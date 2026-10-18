"""Locate the safe once per video, then score open/closed over time with SigLIP on an expanded crop.
usage: probe_safe2.py out_dir name [name ...]"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np, torch
from PIL import Image
from service import signals as S

SAFE_Q = ["a grey metal safe", "a closed metal cabinet", "a small grey metal box"]
STATE = ["a grey metal safe with its door open", "a closed grey metal safe"]
m = S._load()
with torch.no_grad():
    ti = m["sp"](text=STATE, padding="max_length", return_tensors="pt").to(S.DEV)
    tf = m["sm"].get_text_features(**ti); tf = getattr(tf, "pooler_output", tf); tf = tf / tf.norm(dim=-1, keepdim=True)


@torch.no_grad()
def state(crop):
    ii = m["sp"](images=Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), return_tensors="pt").to(S.DEV)
    f = m["sm"].get_image_features(pixel_values=ii["pixel_values"].to(S.DT)); f = getattr(f, "pooler_output", f)
    f = f / f.norm(dim=-1, keepdim=True)
    return torch.softmax((f @ tf.T)[0].float() * m["sm"].logit_scale.exp().float(), -1)[0].item()


def locate(path, desk, H, W):
    x0, y0, x1, y1 = desk; w, h = x1 - x0, y1 - y0
    X0, Y0 = int(max(0, x0 - 0.4 * w)), int(max(0, y0 - 0.9 * h))
    X1, Y1 = int(min(W, x1 + 0.4 * w)), int(min(H, y1 + 0.2 * h))
    best = (0, None)
    for t in (0.5, 3, 6):
        fr = S.read_frame(path, t)
        _, prob, boxes = S._owl(fr[Y0:Y1, X0:X1], SAFE_Q, 0.0)
        p = prob.max(-1).values
        i = int(p.argmax())
        if float(p[i]) > best[0]:
            b = boxes[i].tolist(); best = (float(p[i]), [b[0] + X0, b[1] + Y0, b[2] + X0, b[3] + Y0])
    return best


out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for name in sys.argv[2:]:
    sig = json.load(open(f"runtime/signals/{name}.json", encoding="utf-8"))
    path = [p for p in glob.glob("open/open/videos/*.mov") if os.path.basename(p).replace("Х", "X")[:-4] == name][0]
    H, W = sig["height"], sig["width"]
    sc, b = locate(path, sig["layout"]["desk"]["box"], H, W)
    bw, bh = b[2] - b[0], b[3] - b[1]
    E = [int(max(0, b[0] - 0.6 * bw)), int(max(0, b[1] - 0.15 * bh)), int(min(W, b[2] + 0.6 * bw)), int(min(H, b[3] + 0.15 * bh))]
    series, tiles = [], []
    for t in np.arange(0, sig["duration"] - 0.2, 1.0):
        fr = S.read_frame(path, t)
        c = fr[E[1]:E[3], E[0]:E[2]]
        p = state(c); series.append(f"{t:.0f}:{p:.2f}")
        if int(t) % 3 == 0 and len(tiles) < 10:
            tt = cv2.resize(c, (140, int(140 * c.shape[0] / c.shape[1])))
            cv2.putText(tt, f"{t:.0f} {p:.2f}", (2, 14), 0, 0.45, (0, 0, 255), 1); tiles.append(tt)
    hh = max(x.shape[0] for x in tiles)
    tiles = [np.vstack([x, np.zeros((hh - x.shape[0], 140, 3), np.uint8)]) for x in tiles]
    open(os.path.join(out, name + ".jpg"), "wb").write(cv2.imencode(".jpg", np.hstack(tiles))[1].tobytes())
    print(name, f"loc={sc:.2f}", [int(v) for v in b], " ".join(series), flush=True)
