"""Run the targeted VLM checks for every training video and cache them: runtime/vlm/<name>.json"""
import sys, os, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.pipeline import vlm_checks
from service.rules import evaluate

vids = sorted(glob.glob(os.path.join(sys.argv[1], "*.mov")))
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
for v in vids:
    name = os.path.splitext(os.path.basename(v))[0].replace("Х", "X")
    sig = json.load(open(f"runtime/signals/{name}.json", encoding="utf-8"))
    t = time.time()
    ev, log = vlm_checks(v, sig, evaluate(sig))
    json.dump(dict(events=ev, log=log), open(os.path.join(out, name + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{name} {time.time() - t:.1f}s {log.get('door', '-')}", flush=True)
