"""Cache per-frame id/money detector scores for every training video: runtime/idcard/<name>.json"""
import sys, os, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import idcard

os.makedirs("runtime/idcard", exist_ok=True)
for v in sorted(glob.glob("open/open/videos/*.mov")):
    name = os.path.basename(v).replace("Х", "X")[:-4]
    sig = json.load(open(f"runtime/signals/{name}.json", encoding="utf-8"))
    t = time.time(); idcard.annotate(v, sig)
    json.dump([[f["t"], f.get("id"), f.get("money"), f.get("key"), f.get("unlocked")] for f in sig["frames"]], open(f"runtime/idcard/{name}.json", "w"))
    print(name, f"{time.time() - t:.1f}s", flush=True)
