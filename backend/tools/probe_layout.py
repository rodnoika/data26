"""Locate static layout (desk, safe, laptop, door) per video with OWLv2 on a few frames.
usage: probe_layout.py out.jpg videos_dir"""
import sys, os, glob
import cv2, torch, numpy as np
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

Q = ["a wooden office desk", "a grey metal safe", "a laptop", "a glass door with a handle", "a banknote counting machine"]
COL = [(0, 200, 0), (0, 0, 255), (255, 0, 0), (0, 200, 255), (255, 0, 255)]
dev = "cuda"
op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
om = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", torch_dtype=torch.float16).to(dev).eval()


def owl(img):
    inp = op(text=[Q], images=img, return_tensors="pt").to(dev); inp["pixel_values"] = inp["pixel_values"].half()
    with torch.no_grad():
        o = om(**inp)
    S = max(img.size)
    prob = torch.sigmoid(o.logits[0].float())  # N, Q
    cx, cy, w, h = (o.pred_boxes[0].float() * S).unbind(-1)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
    best = {}
    for q in range(len(Q)):
        i = int(prob[:, q].argmax()); best[q] = (float(prob[i, q]), boxes[i].tolist())
    return best


tiles = []
for f in sorted(glob.glob(os.path.join(sys.argv[2], "*.mov"))):
    cap = cv2.VideoCapture(f); cap.set(cv2.CAP_PROP_POS_MSEC, 500); ok, fr = cap.read()
    best = owl(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    vis = fr.copy()
    for q, (s, b) in best.items():
        cv2.rectangle(vis, tuple(map(int, b[:2])), tuple(map(int, b[2:])), COL[q], 2)
        cv2.putText(vis, f"{q}:{s:.2f}", (int(b[0]) + 3, int(b[1]) + 18), 0, 0.6, COL[q], 2)
    cv2.putText(vis, os.path.basename(f).replace("Х", "X"), (5, 830), 0, 0.9, (0, 0, 0), 2)
    tiles.append(cv2.resize(vis, (232, 424)))
cols = 9
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
img = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
open(sys.argv[1], "wb").write(cv2.imencode(".jpg", img, [1, 85])[1].tobytes())
