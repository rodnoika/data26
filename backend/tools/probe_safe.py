"""Probe safe localisation/state: OWLv2 on the desk neighbourhood. usage: probe_safe.py out.jpg name:t ..."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from service import signals as S

Q = ["a grey metal safe", "a safe with an open door", "an open metal door", "a closed metal cabinet"]
tiles = []
for spec in sys.argv[2:]:
    name, t = spec.split(":"); t = float(t)
    sig = json.load(open(f"runtime/signals_v1/{name}.json", encoding="utf-8"))
    path = [p for p in glob.glob("open/open/videos/*.mov") if os.path.basename(p).replace("Х", "X")[:-4] == name][0]
    fr = S.read_frame(path, t); H, W = fr.shape[:2]
    x0, y0, x1, y1 = sig["layout"]["desk"]["box"]
    w, h = x1 - x0, y1 - y0
    X0, Y0, X1, Y1 = int(max(0, x0 - 0.4 * w)), int(max(0, y0 - 0.9 * h)), int(min(W, x1 + 0.4 * w)), int(min(H, y1 + 0.2 * h))
    crop = fr[Y0:Y1, X0:X1].copy()
    dets, prob, boxes = S._owl(crop, Q, 0.0)
    for q, col in zip(range(len(Q)), [(0, 0, 255), (0, 200, 0), (255, 0, 0), (200, 0, 200)]):
        i = int(prob[:, q].argmax()); b = boxes[i].tolist()
        cv2.rectangle(crop, tuple(map(int, b[:2])), tuple(map(int, b[2:])), col, 2)
        cv2.putText(crop, f"{q}:{float(prob[i, q]):.2f}", (int(b[0]) + 2, int(b[1]) + 14 + 14 * q), 0, 0.45, col, 1)
    cv2.putText(crop, f"{name}@{t}", (3, 14), 0, 0.5, (0, 0, 0), 1)
    tiles.append(cv2.resize(crop, (300, int(300 * crop.shape[0] / crop.shape[1]))))
h = max(t.shape[0] for t in tiles)
tiles = [np.vstack([t, np.zeros((h - t.shape[0], 300, 3), np.uint8)]) for t in tiles]
open(sys.argv[1], "wb").write(cv2.imencode(".jpg", np.hstack(tiles))[1].tobytes())
