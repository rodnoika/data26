"""Per client session: ask the VLM, on zoomed crops around the cashier, whether she holds an ID card.
usage: probe_idcheck.py name ..."""
import sys, os, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from PIL import Image
from service import rules as R
from service.signals import read_frame
from probe_vlm_crop import ask_image

Q = ("Close-up from a bank CCTV camera. Is a person holding a small white ID card or passport in their hand "
     "(not banknotes, not a phone)? Answer 'yes' or 'no', then a few words.")


def sessions(sig):
    """Start times of client sessions: 2 people seated, stable."""
    fr, H = sig["frames"], sig["height"]
    R.evaluate(sig)
    ns = R._smooth([len(R._seated(f, H)) for f in fr], 3)
    out, prev = [], 0
    for i, n in enumerate(ns):
        if n >= 2 and prev < 2:
            out.append(i)
        prev = n
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        sig = json.load(open(f"runtime/signals/{name}.json", encoding="utf-8"))
        path = [p for p in glob.glob("open/open/videos/*.mov") if os.path.basename(p).replace("Х", "X")[:-4] == name][0]
        fr, H, W = sig["frames"], sig["height"], sig["width"]
        for i0 in sessions(sig):
            t0 = fr[i0]["t"]; line = []
            for f in fr[i0:]:
                if f["t"] > t0 + 12:
                    break
                if int(f["t"] * 2) % 2:          # 1 frame / s
                    continue
                st = R._seated(f, H)
                if not st:
                    continue
                x0 = min(b[0] for b in st); y0 = min(b[1] for b in st); x1 = max(b[2] for b in st); y1 = max(b[3] for b in st)
                full = read_frame(path, f["t"])
                if full is None:
                    continue
                img = full[int(max(0, y0)):int(min(H, y1)), int(max(0, x0 - 20)):int(min(W, x1 + 20))]
                img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                a = ask_image(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), Q)
                line.append(f"{f['t']:.0f}:{'Y' if a.lower().startswith('yes') else '.'}")
            print(name, f"session@{t0:.0f}", " ".join(line), flush=True)
