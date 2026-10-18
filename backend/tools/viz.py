"""Draw cached detections on frames. usage: viz.py out.jpg name:t [name:t ...]   (needs runtime/signals)"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from service import rules as R

VID = "open/open/videos"
tiles = []
for spec in sys.argv[2:]:
    name, t = spec.split(":"); t = float(t)
    sig = json.load(open(f"runtime/signals/{name}.json", encoding="utf-8"))
    R._CTX["desk_top"] = sig["layout"]["desk"]["box"][1]
    f = min(sig["frames"], key=lambda f: abs(f["t"] - t))
    path = [p for p in glob.glob(VID + "/*.mov") if os.path.basename(p).replace("Х", "X")[:-4] == name][0]
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_MSEC, f["t"] * 1000); ok, fr = cap.read()
    x0, y0, x1, y1 = map(int, sig["layout"]["desk"]["box"]); cv2.rectangle(fr, (x0, y0), (x1, y1), (0, 200, 0), 1)
    inroom = [tuple(b) for b in R._people(f, sig["height"])]
    for n, s, b in f["coco"]:
        if n == "person":
            c = (0, 0, 255) if tuple(b) in inroom else (180, 180, 180)
        elif n in ("cup", "bottle", "handbag", "laptop"):
            c = (255, 0, 0)
        else:
            continue
        cv2.rectangle(fr, tuple(map(int, b[:2])), tuple(map(int, b[2:])), c, 2)
        cv2.putText(fr, f"{n[:6]} {s:.2f}", (int(b[0]), int(b[1]) + 14), 0, 0.45, c, 1)
    for g, s, b in f["desk"]:
        if s > 0.2:
            cv2.rectangle(fr, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 220, 220), 1)
            cv2.putText(fr, f"{g} {s:.2f}", (int(b[0]), int(b[3]) - 2), 0, 0.4, (0, 160, 160), 1)
    cv2.putText(fr, f"{name} {f['t']}s scr={f['screen']}", (5, 20), 0, 0.6, (0, 0, 0), 2)
    tiles.append(fr[:620])
img = np.hstack(tiles)
open(sys.argv[1], "wb").write(cv2.imencode(".jpg", img, [1, 85])[1].tobytes())
