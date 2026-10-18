"""Locate the banknote detector device (OWLv2) and print how far the cashier's box reaches toward it over time.
usage: probe_detdev.py name ..."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import signals as S
from service import rules as R

Q = ["a small black banknote detector device", "a black electronic device with a light", "a counterfeit money detector"]


def locate(path, sig):
    x0, y0, x1, y1 = sig["layout"]["desk"]["box"]
    best = (0, None)
    for t in (0.5, sig["duration"] * 0.5):
        fr = S.read_frame(path, t)
        crop = fr[int(y0 - 60):int(y1), int(x0):int(x1)]
        _, prob, boxes = S._owl(crop, Q, 1.0)
        pm = prob.max(-1).values; i = int(pm.argmax())
        if float(pm[i]) > best[0]:
            b = boxes[i].tolist(); best = (float(pm[i]), [b[0] + x0, b[1] + y0 - 60, b[2] + x0, b[3] + y0 - 60])
    return best


if __name__ == "__main__":
    for name in sys.argv[1:]:
        sig = json.load(open(f"runtime/signals/{name}.json", encoding="utf-8"))
        path = [p for p in glob.glob("open/open/videos/*.mov") if os.path.basename(p).replace("Х", "X")[:-4] == name][0]
        sc, b = locate(path, sig)
        d = sig["layout"]["desk"]["box"]; R._CTX.update(desk=d, desk_top=d[1], W=sig["width"])
        cnt = sig["layout"].get("counter", {}).get("box")
        near = []
        for f in sig["frames"]:
            st = R._seated(f, sig["height"])
            if not st:
                continue
            c = max(st, key=lambda q: q[3])
            ix = max(0, min(c[2], b[2]) - max(c[0], b[0])) * max(0, min(c[3], b[3]) - max(c[1], b[1]))
            mon = any(g == "money" and s >= .2 and R._iou(bb, b) > 0 for g, s, bb in f["desk"])
            if ix > 0.1 * (b[2] - b[0]) * (b[3] - b[1]) or mon:
                near.append(f"{f['t']:.0f}{'$' if mon else ''}")
        print(name, f"dev={sc:.2f}", [int(v) for v in b], "near:", " ".join(near))
