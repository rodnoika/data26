"""Door-left-open and standing-outsider probes.
door: patch where the first person appears; compare its look before entry vs. after the cashier sat down.
usage: probe_door.py out_dir name [name ...]"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from service import rules as R
from service.signals import read_frame

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
tiles = []
for name in sys.argv[2:]:
    sig = json.load(open(f"runtime/signals/{name}.json", encoding="utf-8"))
    H = sig["height"]; d = sig["layout"]["desk"]["box"]; R._CTX.update(desk=d, desk_top=d[1])
    path = [p for p in glob.glob("open/open/videos/*.mov") if os.path.basename(p).replace("Х", "X")[:-4] == name][0]
    fr = sig["frames"]
    # first frame with any confident tall person = entry
    ent = next((f for f in fr if any(n == "person" and s > .8 and b[3] - b[1] > .17 * H for n, s, b in f["coco"])), None)
    seat = next((f for f in fr if R._seated(f, H) and ent and f["t"] > ent["t"]), None)
    if not ent or not seat or ent["t"] < 0.6:
        print(name, "no entry", flush=True); continue
    b = max([b for n, s, b in ent["coco"] if n == "person" and s > .8], key=lambda b: b[3] - b[1])
    x0, y0, x1, y1 = [int(v) for v in b]; pad = int(0.3 * (x1 - x0))
    x0, x1 = max(0, x0 - pad), min(sig["width"], x1 + pad)
    before = read_frame(path, max(0, ent["t"] - 0.8))[y0:y1, x0:x1]
    after_t = min(sig["duration"] - 0.2, seat["t"] + 3)
    after = read_frame(path, after_t)[y0:y1, x0:x1]
    g = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    diff = float(np.abs(g(before) - g(after)).mean())
    # is a person in the patch in the "after" frame? (then the diff is meaningless)
    fa = min(fr, key=lambda f: abs(f["t"] - after_t))
    occ = any(n == "person" and s > .5 and not (bb[2] < x0 or bb[0] > x1 or bb[3] < y0 or bb[1] > y1) and not R._seated({"coco": [(n, s, bb)]}, H)
              for n, s, bb in fa["coco"])
    print(name, f"entry@{ent['t']} seated@{seat['t']} diff={diff:.1f} occluded={occ}", flush=True)
    t = np.hstack([before, after]); t = cv2.resize(t, (240, int(240 * t.shape[0] / t.shape[1])))
    cv2.putText(t, f"{name} {diff:.0f}", (2, 14), 0, 0.5, (0, 0, 255), 1); tiles.append(t)
hh = max(x.shape[0] for x in tiles)
tiles = [np.vstack([x, np.zeros((hh - x.shape[0], 240, 3), np.uint8)]) for x in tiles]
open(os.path.join(out, "door.jpg"), "wb").write(cv2.imencode(".jpg", np.hstack(tiles))[1].tobytes())
