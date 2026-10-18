"""Per-frame signal extraction (the only part that runs neural nets).

For every sampled frame we record: people in the room, COCO objects of interest, objects lying on
the desk (OWLv2 on the upscaled desk crop), and laptop-screen brightness. Rules work on these.
"""
import os
import cv2, numpy as np, torch
from PIL import Image

os.environ.setdefault("HF_HOME", "C:/hf_cache")  # sentencepiece/tokenizers break on non-ASCII paths
from transformers import (RTDetrImageProcessor, RTDetrV2ForObjectDetection, Owlv2Processor,
                          Owlv2ForObjectDetection, AutoProcessor, AutoModel)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16 if DEV == "cuda" else torch.float32
STEP_SEC = 0.5

LAYOUT_Q = ["a wooden office desk", "a laptop", "a banknote counting machine"]
DESK_Q = ["banknotes", "paper money", "a stack of banknotes", "an id card",   # 0-3 money-like
          "a green bag",                                                     # 4 cash-in-transit bag
          "a handbag", "a jacket",                                           # 5-6 personal belongings
          "a black folder", "documents",                                     # 7-8 documents
          "a cup", "a bottle"]                                               # 9-10 drinks
DESK_GROUP = {0: "money", 1: "money", 2: "money", 3: "money", 4: "bag", 5: "personal", 6: "personal",
              7: "docs", 8: "docs", 9: "drink", 10: "drink"}
SCREEN_PROMPTS = ["a laptop screen showing a spreadsheet or an open application",
                  "a laptop with a lock screen wallpaper", "a laptop with a black turned off screen"]
DETECTOR_Q = ["a small black banknote detector device", "a black electronic device with a light",
              "a counterfeit money detector"]
SAFE_Q = ["a grey metal safe", "a closed metal cabinet", "a small grey metal box"]
SAFE_PROMPTS = ["a grey metal safe with its door open", "a closed grey metal safe"]
COCO_KEEP = {"person", "cup", "bottle", "handbag", "backpack", "suitcase", "laptop", "cell phone", "wine glass"}

_models = {}


def _load():
    if not _models:
        rp = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd")
        rm = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd").to(DEV).eval().to(DT)
        op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        om = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", torch_dtype=DT).to(DEV).eval()
        sp = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        sm = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=DT).to(DEV).eval()
        with torch.no_grad():
            ti = sp(text=SCREEN_PROMPTS, padding="max_length", return_tensors="pt").to(DEV)
            tf = sm.get_text_features(**ti); tf = getattr(tf, "pooler_output", tf)
            ti = sp(text=SAFE_PROMPTS, padding="max_length", return_tensors="pt").to(DEV)
            sf = sm.get_text_features(**ti); sf = getattr(sf, "pooler_output", sf)
        _models.update(rp=rp, rm=rm, op=op, om=om, sp=sp, sm=sm, screen_txt=tf / tf.norm(dim=-1, keepdim=True),
                       safe_txt=sf / sf.norm(dim=-1, keepdim=True))
    return _models


@torch.no_grad()
def _owl(img_bgr, queries, thr):
    m = _load()
    if img_bgr is None or img_bgr.size == 0:          # crop fell outside the frame
        z = torch.zeros((1, len(queries)), device=DEV)
        return [], z, torch.zeros((1, 4), device=DEV)
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inp = m["op"](text=[queries], images=img, return_tensors="pt").to(DEV)
    inp["pixel_values"] = inp["pixel_values"].to(DT)
    o = m["om"](**inp)
    S = max(img.size)  # OWLv2 pads to a square; boxes are relative to it
    prob = torch.sigmoid(o.logits[0].float())
    sc, lab = prob.max(-1)
    cx, cy, w, h = (o.pred_boxes[0].float() * S).unbind(-1)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
    keep = sc > thr
    return [(int(l), float(s), [round(v, 1) for v in b])
            for l, s, b in zip(lab[keep].tolist(), sc[keep].tolist(), boxes[keep].tolist())], prob, boxes


@torch.no_grad()
def _rtdetr(img_bgr, thr=0.35):
    m = _load()
    H, W = img_bgr.shape[:2]
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pv = m["rp"](images=img, return_tensors="pt")["pixel_values"].to(DEV).to(DT)
    o = m["rm"](pixel_values=pv)
    r = m["rp"].post_process_object_detection(o, threshold=thr, target_sizes=[(H, W)])[0]
    L = m["rm"].config.id2label
    out = []
    for l, s, b in zip(r["labels"].tolist(), r["scores"].tolist(), r["boxes"].tolist()):
        n = L[l]
        if n in COCO_KEEP:
            out.append((n, round(float(s), 3), [round(v, 1) for v in b]))
    return out


def _layout(cap, dur):
    """Static scene layout: desk / laptop / counter boxes, voted over a few frames."""
    votes = {q: [] for q in range(len(LAYOUT_Q))}
    for t in (0.5, dur * 0.33, dur * 0.66):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        _, prob, boxes = _owl(fr, LAYOUT_Q, 1.0)
        for q in votes:
            i = int(prob[:, q].argmax())
            votes[q].append((float(prob[i, q]), boxes[i].tolist()))
    lay = {}
    for q, name in enumerate(["desk", "laptop", "counter"]):
        if votes[q]:
            s, b = max(votes[q])
            lay[name] = dict(score=round(s, 3), box=[round(v, 1) for v in b])
    return lay


@torch.no_grad()
def _siglip_softmax(crop, key):
    m = _load()
    if crop is None or crop.size == 0:
        return None
    ii = m["sp"](images=Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), return_tensors="pt").to(DEV)
    f = m["sm"].get_image_features(pixel_values=ii["pixel_values"].to(DT)); f = getattr(f, "pooler_output", f)
    f = f / f.norm(dim=-1, keepdim=True)
    logits = (f @ m[key].T)[0].float() * m["sm"].logit_scale.exp().float()
    return [round(v, 3) for v in torch.softmax(logits, -1).tolist()]


def _crop(fr, box):
    x0, y0, x1, y1 = [int(max(0, v)) for v in box]
    return fr[y0:y1, x0:x1]


def screen_state(fr, box):
    """SigLIP zero-shot on the laptop crop -> probabilities [unlocked app, lock screen, off]."""
    return _siglip_softmax(_crop(fr, box), "screen_txt")


def safe_state(fr, box):
    """SigLIP zero-shot on the expanded safe crop -> P(open)."""
    p = _siglip_softmax(_crop(fr, box), "safe_txt")
    return p[0] if p else None


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix + 1e-6)


def _locate_detector(cap, desk, dur, counter=None):
    """Banknote detector device (small black box with a light next to the counter): best OWLv2 box on the desk
    that is not the counting machine itself (zero-shot OWLv2 often puts both queries on the same object)."""
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    x0, y0, x1, y1 = int(max(0, desk[0])), int(max(0, desk[1] - 60)), int(min(W, desk[2])), int(min(H, desk[3]))
    best = (0.0, None)
    if x1 - x0 < 8 or y1 - y0 < 8:                    # desk box outside the frame (unusual camera angle)
        return None
    for t in (0.5, dur * 0.5):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        _, prob, boxes = _owl(fr[y0:y1, x0:x1], DETECTOR_Q, 1.0)
        pm = prob.max(-1).values
        for i in pm.argsort(descending=True)[:20].tolist():
            b = boxes[i].tolist(); b = [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]
            if counter and _iou(b, counter) >= 0.5:
                continue
            if float(pm[i]) > best[0]:
                best = (float(pm[i]), b)
            break
    return dict(score=round(best[0], 3), box=[round(v, 1) for v in best[1]]) if best[1] else None


def _locate_safe(cap, desk, H, W):
    """Safe = best OWLv2 'metal safe/cabinet' box in the desk neighbourhood over a few frames.
    Returns the expanded box used for the state classifier (an open door swings out sideways)."""
    x0, y0, x1, y1 = desk; w, h = x1 - x0, y1 - y0
    X0, Y0 = int(max(0, x0 - 0.4 * w)), int(max(0, y0 - 0.9 * h))
    X1, Y1 = int(min(W, x1 + 0.4 * w)), int(min(H, y1 + 0.2 * h))
    best = (0.0, None)
    for t in (0.5, 3, 6):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        _, prob, boxes = _owl(fr[Y0:Y1, X0:X1], SAFE_Q, 1.0)
        pm = prob.max(-1).values; i = int(pm.argmax())
        if float(pm[i]) > best[0]:
            b = boxes[i].tolist(); best = (float(pm[i]), [b[0] + X0, b[1] + Y0, b[2] + X0, b[3] + Y0])
    if best[1] is None:
        return None
    b = best[1]; bw, bh = b[2] - b[0], b[3] - b[1]
    return dict(score=round(best[0], 3), box=[round(v, 1) for v in b],
                state_box=[max(0, b[0] - 0.6 * bw), max(0, b[1] - 0.15 * bh), min(W, b[2] + 0.6 * bw), min(H, b[3] + 0.15 * bh)])


def screen_brightness(fr, box):
    """Fraction of near-white pixels in the laptop box: high = open app (spreadsheet), low = lock screen/off."""
    x0, y0, x1, y1 = [int(max(0, v)) for v in box]
    c = fr[y0:y1, x0:x1]
    if c.size == 0:
        return 0.0
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    return round(float((g > 200).mean()), 3)


def extract(path, progress=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = n / fps if n > 0 else 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    lay = _layout(cap, dur)
    desk = lay.get("desk", {}).get("box", [0, H * 0.3, W, H * 0.7])
    safe = _locate_safe(cap, desk, H, W)
    if safe:
        lay["safe"] = safe
    det = _locate_detector(cap, desk, dur, lay.get("counter", {}).get("box"))
    if det:
        lay["detector"] = det
    dx0, dy0, dx1, dy1 = desk
    # include things standing on the desk (cups, bags) above its top edge
    cy0 = max(0, int(dy0 - 0.08 * H)); cy1 = min(H, int(dy1)); cx0 = max(0, int(dx0)); cx1 = min(W, int(dx1))

    frames = []
    cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    next_t, idx = 0.0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = idx / fps; idx += 1
        if t + 1e-6 < next_t:
            continue
        next_t += STEP_SEC
        ok, fr = cap.retrieve()
        if not ok:
            break
        coco = _rtdetr(fr)
        dets, _, _ = _owl(fr[cy0:cy1, cx0:cx1], DESK_Q, 0.15)
        desk_items = [(DESK_GROUP[l], s, [b[0] + cx0, b[1] + cy0, b[2] + cx0, b[3] + cy0]) for l, s, b in dets]
        laptops = [d for d in coco if d[0] == "laptop"]
        lap_box = max(laptops, key=lambda d: d[1])[2] if laptops else lay.get("laptop", {}).get("box")
        frames.append(dict(t=round(t, 2), coco=coco, desk=desk_items,
                           screen=screen_brightness(fr, lap_box) if lap_box else None,
                           screen_p=screen_state(fr, lap_box) if lap_box else None,
                           safe_open=safe_state(fr, safe["state_box"]) if safe else None,
                           laptop=lap_box))
        if progress and dur:
            progress(min(0.95, t / dur))
    cap.release()
    return dict(fps=fps, duration=round(dur, 2), width=W, height=H, layout=lay, frames=frames)


def read_frame(path, t):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None
