"""Quick check of OWLv2 zero-shot detection quality on a few frames.

usage: python tools/probe_owl.py out_dir video:sec [video:sec ...]
"""
import sys, os, time
import cv2, numpy as np, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

QUERIES = ["a person", "banknotes", "a stack of money", "a green fabric bag", "a metal safe",
           "an open safe door", "a laptop", "a cup", "a handbag", "a jacket", "an id card",
           "a banknote counting machine", "a glass door"]

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
dev = "cuda"
name = "google/owlv2-base-patch16-ensemble"
proc = Owlv2Processor.from_pretrained(name)
model = Owlv2ForObjectDetection.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()

for spec in sys.argv[2:]:
    path, sec = spec.rsplit(":", 1)
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000)
    ok, fr = cap.read()
    img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    t = time.time()
    inp = proc(text=[QUERIES], images=img, return_tensors="pt").to(dev)
    inp["pixel_values"] = inp["pixel_values"].half()
    with torch.no_grad():
        o = model(**inp)
    # OWLv2 pads to square: boxes are relative to the padded square image
    S = max(img.size)
    prob = torch.sigmoid(o.logits[0].float())          # (N, Q)
    sc, lab = prob.max(-1)
    keep = sc > 0.12
    cx, cy, w, h = (o.pred_boxes[0][keep].float() * S).unbind(-1)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
    res = {"boxes": boxes, "scores": sc[keep], "labels": lab[keep]}
    dt = time.time() - t
    vis = fr.copy()
    for b, s, l in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist()):
        x0, y0, x1, y1 = map(int, b)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.putText(vis, f"{QUERIES[l][:14]} {s:.2f}", (x0, max(10, y0 - 2)), 0, 0.35, (0, 0, 255), 1)
    tag = os.path.basename(path).replace("Х", "X").replace(".mov", "") + f"_{sec}"
    open(os.path.join(out, tag + ".jpg"), "wb").write(cv2.imencode(".jpg", vis)[1].tobytes())
    print(tag, f"{dt*1000:.0f}ms", len(res["scores"]), flush=True)
