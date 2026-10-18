"""Cash flow between cashier and client: position of moving cash boxes projected on the cashier->client axis.
s=0 at the cashier's box centre, s=1 at the client's. Static boxes (ID card lying on the desk) are dropped.
usage: probe_cashflow.py name [name ...]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import rules as R


def flow(sig):
    H = sig["height"]; d = sig["layout"]["desk"]["box"]
    R._CTX.update(desk=d, desk_top=d[1], W=sig["width"])
    fr = sig["frames"]
    boxes = [b for f in fr for g, s, b in f["desk"] if g == "money" and s >= 0.2]
    def static(b):
        return sum(R._iou(b, o) > 0.7 for o in boxes) >= 6
    out = []
    for f in fr:
        seated = R._seated(f, H)
        if len(seated) < 2:
            continue
        cash = max(seated, key=lambda b: b[3])          # the person nearer the camera bottom... see note
        cli = min(seated, key=lambda b: b[3])
        # cashier = the person who stays seated the whole video; approximate: larger box bottom (near side)
        c0 = ((cash[0] + cash[2]) / 2, (cash[1] + cash[3]) / 2); c1 = ((cli[0] + cli[2]) / 2, (cli[1] + cli[3]) / 2)
        vx, vy = c1[0] - c0[0], c1[1] - c0[1]; L2 = vx * vx + vy * vy or 1
        for g, s, b in f["desk"]:
            if g == "money" and s >= 0.2 and not static(b):
                m = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                out.append((f["t"], round(((m[0] - c0[0]) * vx + (m[1] - c0[1]) * vy) / L2, 2), round(s, 2)))
    return out


for name in sys.argv[1:]:
    sig = json.load(open(f"runtime/signals/{name}.json", encoding="utf-8"))
    print(name, " ".join(f"{t:.1f}:{p}" for t, p, s in flow(sig)))
