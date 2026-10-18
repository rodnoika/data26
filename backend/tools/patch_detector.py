"""Add layout["detector"] to cached signal files without re-running the per-frame models."""
import sys, os, glob, json
# usage: patch_detector.py [signals_dir videos_dir]   (default: runtime/signals, open/open/videos)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from service.signals import _locate_detector

sd, vd = (sys.argv[1:3] if len(sys.argv) > 2 else ("runtime/signals", "open/open/videos"))
for v in sorted(glob.glob(f"{vd}/*.mov") + glob.glob(f"{vd}/*.mp4")):
    name = os.path.basename(v).replace("Х", "X")[:-4]
    p = f"{sd}/{name}.json"
    if not os.path.exists(p):
        continue
    sig = json.load(open(p, encoding="utf-8"))
    cap = cv2.VideoCapture(v)
    sig["layout"]["detector"] = _locate_detector(cap, sig["layout"]["desk"]["box"], sig["duration"],
                                                 sig["layout"].get("counter", {}).get("box"))
    json.dump(sig, open(p, "w", encoding="utf-8"))
    print(name, sig["layout"]["detector"], flush=True)
