"""Cache raw signals for every training video: runtime/signals/<name>.json"""
import sys, os, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.signals import extract

vids = sorted(glob.glob(os.path.join(sys.argv[1], "*.mov")))
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
for v in vids:
    name = os.path.splitext(os.path.basename(v))[0].replace("Х", "X")
    dst = os.path.join(out, name + ".json")
    if os.path.exists(dst):
        continue
    t = time.time()
    sig = extract(v)
    json.dump(sig, open(dst, "w", encoding="utf-8"))
    print(f"{name} dur={sig['duration']}s took={time.time() - t:.1f}s", flush=True)
