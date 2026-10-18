"""Robustness check: re-run the whole pipeline on geometrically altered copies of the training videos.

usage: augment_eval.py <variant> [<variant> ...]      variants: flip, warp, zoom
Writes runtime/aug/<variant>/{videos,signals,idcard,vlm}/ so backend/tools/evaluate.py can score it:
    python backend/tools/evaluate.py runtime/aug/<variant>/signals open/open/open_labels.xlsx violation_classification.csv
Each stage is cached per video, so the script can be restarted.
"""
import sys, os, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from service.signals import extract
from service import idcard
from service.rules import evaluate
from service.pipeline import vlm_checks

SKIP = {"X-47", "X-48"}   # byte-identical duplicates of Х-4 / Х-5


def transform(variant, W, H):
    if variant == "flip":
        return lambda fr: cv2.flip(fr, 1)
    if variant == "warp":   # camera turned ~10-15 degrees: one side of the image shrinks
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = np.float32([[0.10 * W, 0.04 * H], [W, 0], [0.94 * W, H], [0, 0.93 * H]])
        M = cv2.getPerspectiveTransform(src, dst)
        return lambda fr: cv2.warpPerspective(fr, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    if variant == "zoom":   # camera closer and shifted: off-centre 80 % crop
        x0, y0, x1, y1 = int(0.12 * W), int(0.06 * H), int(0.92 * W), int(0.86 * H)
        return lambda fr: cv2.resize(fr[y0:y1, x0:x1], (W, H))
    raise ValueError(variant)


def write_video(src, dst, variant):
    cap = cv2.VideoCapture(src)
    fps, W, H = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    f = transform(variant, W, H)
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.write(f(fr))
    out.release(); cap.release()


for variant in sys.argv[1:]:
    root = f"runtime/aug/{variant}"
    for d in ("videos", "signals", "idcard", "vlm"):
        os.makedirs(f"{root}/{d}", exist_ok=True)
    for v in sorted(glob.glob("open/open/videos/*.mov")):
        name = os.path.basename(v).replace("Х", "X")[:-4]
        if name in SKIP:
            continue
        t = time.time()
        vp = f"{root}/videos/{name}.mp4"
        if not os.path.exists(vp):
            write_video(v, vp, variant)
        sp = f"{root}/signals/{name}.json"
        if not os.path.exists(sp):
            json.dump(extract(vp), open(sp, "w", encoding="utf-8"))
        sig = json.load(open(sp, encoding="utf-8"))
        ip = f"{root}/idcard/{name}.json"
        if not os.path.exists(ip):
            idcard.annotate(vp, sig)
            json.dump([[f["t"], f.get("id"), f.get("money"), f.get("key")] for f in sig["frames"]], open(ip, "w"))
        qp = f"{root}/vlm/{name}.json"
        if not os.path.exists(qp):
            sc = {round(r[0], 2): r[1:] for r in json.load(open(ip))}
            for fr_ in sig["frames"]:
                fr_["id"], fr_["money"], fr_["key"] = sc.get(round(fr_["t"], 2), (None, None, None))
            ev, log = vlm_checks(vp, sig, evaluate(sig))
            json.dump(dict(events=ev, log=log), open(qp, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"{variant} {name} {time.time() - t:.0f}s", flush=True)
