"""Run the fine-tuned ID detector on client sessions of other videos (not used in training).
Prints per second: max score, and saves a strip of the top detection per video.
usage: test_id.py out_dir name ..."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from service import idcard
from service.signals import read_frame

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for name in sys.argv[2:]:
    sig = json.load(open(f"runtime/signals/{name}.json", encoding="utf-8"))
    path = [p for p in glob.glob("open/open/videos/*.mov") if os.path.basename(p).replace("Х", "X")[:-4] == name][0]
    crop = idcard.desk_crop_box(sig["layout"]["desk"]["box"], sig["height"], sig["width"])
    row, best = [], (0, None, None)
    t = 0.0
    while t < min(sig["duration"], 40):
        fr = read_frame(path, t)
        if fr is None:
            break
        d = idcard.detect(fr, crop, thr=0.05)
        s = d[0][0] if d else 0.0
        dm = idcard.detect(fr, crop, thr=0.05, cls="money")
        row.append(f"{t:.0f}:{s:.2f}/{dm[0][0] if dm else 0.0:.2f}")
        if s > best[0]:
            best = (s, t, (fr, d[0][1]))
        t += 1.0
    if best[2]:
        fr, b = best[2]; x0, y0, x1, y1 = map(int, b)
        cv2.rectangle(fr, (x0, y0), (x1, y1), (0, 0, 255), 2)
        c = fr[max(0, y0 - 80):y1 + 80, max(0, x0 - 80):x1 + 80]
        cv2.putText(c, f"{name} {best[1]:.0f}s {best[0]:.2f}", (2, 14), 0, 0.45, (0, 255, 255), 1)
        open(os.path.join(out, name + ".jpg"), "wb").write(cv2.imencode(".jpg", c)[1].tobytes())
    print(name, " ".join(row), flush=True)
