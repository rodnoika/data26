"""ID-card detector: RT-DETRv2-R18 (Apache-2.0) fine-tuned on CVAT labels (tools/train_id.py), run on the desk crop."""
import os
import cv2, numpy as np, torch

SIZE = 640
_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
WEIGHTS = os.environ.get("ID_WEIGHTS", os.path.join(_ROOT, "idmoney_rtdetr"))
KEY_WEIGHTS = os.environ.get("KEY_WEIGHTS", os.path.join(_ROOT, "key_rtdetr"))
LAPTOP_WEIGHTS = os.environ.get("LAPTOP_WEIGHTS", os.path.join(_ROOT, "unlocked_rtdetr"))
_m = {}


def desk_crop_box(desk, H, W):
    """Desk box extended upwards (hands of a standing client / cashier holding the card) and sideways."""
    x0, y0, x1, y1 = desk
    w, h = x1 - x0, y1 - y0
    return [max(0, x0 - 0.1 * w), max(0, y0 - 0.15 * H), min(W, x1 + 0.1 * w), min(H, y1 + 0.05 * h)]


def safe_crop_box(safe, H, W):
    """Safe box enlarged x2 around its centre: the key is a ~10 px stub on the door."""
    x0, y0, x1, y1 = safe
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    return [max(0, cx - w), max(0, cy - h), min(W, cx + w), min(H, cy + h)]


def laptop_crop_box(lap, H, W):
    """Laptop box enlarged x1.6 around its centre."""
    x0, y0, x1, y1 = lap
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 0.8, (y1 - y0) * 0.8
    return [max(0, cx - w), max(0, cy - h), min(W, cx + w), min(H, cy + h)]


def available(weights=None):
    return os.path.isdir(weights or WEIGHTS) and torch.cuda.is_available()


def _load(weights=None):
    w = weights or WEIGHTS
    if w not in _m:
        from transformers import RTDetrV2ForObjectDetection
        _m[w] = RTDetrV2ForObjectDetection.from_pretrained(w).cuda().eval().half()
    return {"model": _m[w]}


@torch.no_grad()
def detect(fr, crop, thr=0.3, cls="id", weights=None):
    """-> [(score, [x0,y0,x1,y1] in frame px)] for one class ("id" or "money" in the two-class model)"""
    m = _load(weights or {"key": KEY_WEIGHTS, "unlocked": LAPTOP_WEIGHTS}.get(cls))
    x0, y0, x1, y1 = [int(v) for v in crop]
    if x1 - x0 < 4 or y1 - y0 < 4:                    # region outside the frame (unusual camera angle)
        return []
    img = cv2.resize(fr[y0:y1, x0:x1], (SIZE, SIZE))
    px = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)[None].cuda().half()
    o = m["model"](pixel_values=px)
    names = {v: int(k) for k, v in m["model"].config.id2label.items()}
    if cls not in names:
        return []
    sc = o.logits[0].float().sigmoid()[:, names[cls]]
    keep = sc > thr
    cx, cy, w, h = o.pred_boxes[0][keep].float().unbind(-1)
    sx, sy = (x1 - x0), (y1 - y0)
    out = []
    for s, a, b, c, d in zip(sc[keep].tolist(), cx.tolist(), cy.tolist(), w.tolist(), h.tolist()):
        out.append((round(s, 3), [x0 + (a - c / 2) * sx, y0 + (b - d / 2) * sy, x0 + (a + c / 2) * sx, y0 + (b + d / 2) * sy]))
    return sorted(out, reverse=True)


def annotate(path, sig, thr=0.3):
    """Add per-frame max 'id' / 'money' (desk crop) and 'key' (safe crop) scores to sig["frames"] (same 0.5 s timestamps as the signals)."""
    crop = desk_crop_box(sig["layout"]["desk"]["box"], sig["height"], sig["width"])
    sf = sig["layout"].get("safe")
    kcrop = safe_crop_box(sf["box"], sig["height"], sig["width"]) if sf and available(KEY_WEIGHTS) else None
    lp = sig["layout"].get("laptop")
    # trained on Х-11 only, it learned that camera, not the screen state (fires on black screens) - off by default
    use_lap = os.environ.get("USE_LAPTOP_DET") == "1" and lp and available(LAPTOP_WEIGHTS)
    lcrop = laptop_crop_box(lp["box"], sig["height"], sig["width"]) if use_lap else None
    cap = cv2.VideoCapture(path)
    for f in sig["frames"]:
        cap.set(cv2.CAP_PROP_POS_MSEC, f["t"] * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        di, dm = detect(fr, crop, thr, "id"), detect(fr, crop, thr, "money")
        f["id"] = [di[0][0], [round(v, 1) for v in di[0][1]]] if di else None
        f["money"] = [dm[0][0], [round(v, 1) for v in dm[0][1]]] if dm else None
        if kcrop:
            dk = detect(fr, kcrop, thr, "key")
            f["key"] = [dk[0][0], [round(v, 1) for v in dk[0][1]]] if dk else None
        if lcrop:
            dl = detect(fr, lcrop, thr, "unlocked")
            f["unlocked"] = [dl[0][0], [round(v, 1) for v in dl[0][1]]] if dl else None
    cap.release()
    return sig
