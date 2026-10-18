"""Score the rules engine on cached signals against open_labels.xlsx and the manual type table.

usage: python backend/tools/evaluate.py runtime/signals open/open/open_labels.xlsx violation_classification.csv
Prints per-video verdict, predicted types/timecodes vs. label, and totals.
Duplicates (Х-47 = Х-4, Х-48 = Х-5) are skipped so they do not double-count.
"""
import sys, os, json, csv, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openpyxl
from service.rules import evaluate, review_hints

SAFETY = os.environ.get("SAFETY_NET", "default")

SKIP = {"X-47", "X-48"}
sig_dir, xlsx, types_csv = sys.argv[1:4]


def norm(n):
    return n.replace("Х", "X").strip()


def secs(s):
    if not s:
        return None
    m = re.findall(r"\d+", str(s))
    return int(m[0]) * 60 + int(m[1]) if "мин" in str(s) and len(m) > 1 else int(m[0])


labels = {}
for row in list(openpyxl.load_workbook(xlsx).active.iter_rows(values_only=True))[1:]:
    if row[0]:
        labels[norm(row[0])] = (row[2].strip() == "нарушение", secs(row[1]))
gt_type = {norm(r["file"]): r["type_id"] for r in csv.DictReader(open(types_csv, encoding="utf-8")) if r["type_id"]}

tp = fp = fn = tn = 0; type_ok = 0; tc_err = []
for name in sorted(labels, key=lambda n: int(re.findall(r"\d+", n)[0])):
    if name in SKIP:
        continue
    p = os.path.join(sig_dir, name + ".json")
    if not os.path.exists(p):
        continue
    sig = json.load(open(p, encoding="utf-8"))
    ip = os.path.join(os.path.dirname(sig_dir.rstrip("/\\")), "idcard", name + ".json")
    if os.path.exists(ip):                       # cached id/money detector scores (backend/tools/idcard_cache.py)
        sc = {round(r[0], 2): (r[1:] + [None])[:3] for r in json.load(open(ip))}
        for f in sig["frames"]:
            f["id"], f["money"], f["key"] = sc.get(round(f["t"], 2), (None, None, None))
    ev = evaluate(sig)
    vp = os.path.join(os.path.dirname(sig_dir.rstrip("/\\")), "vlm", name + ".json")
    if os.path.exists(vp):                       # cached targeted VLM checks (backend/tools/vlm_cache.py)
        ev = ev + json.load(open(vp, encoding="utf-8"))["events"]
    ev = ev + [h for h in review_hints(sig, ev) if h.get("promote") or SAFETY == "verdict"]
    is_v, t_gt = labels[name]
    pred = bool(ev)
    tp += is_v and pred; fp += (not is_v) and pred; fn += is_v and not pred; tn += (not is_v) and not pred
    s = ", ".join(f"{e['type_id']}@{e['t']:.1f}" for e in ev) or "-"
    mark = "OK " if is_v == pred else ("FN " if is_v else "FP ")
    extra = ""
    if is_v and pred:
        best = min(ev, key=lambda e: abs(e["t"] - t_gt))
        tc_err.append(abs(best["t"] - t_gt))
        hit = any(e["type_id"] == gt_type.get(name) for e in ev)
        type_ok += hit
        extra = f"  |dt|={abs(best['t'] - t_gt):.1f}s type={'ok' if hit else 'MISS'}"
    print(f"{mark}{name:6s} gt={'V ' + gt_type.get(name, '?') + '@' + str(t_gt) if is_v else 'clean':12s} pred={s}{extra}")

n = tp + fp + fn + tn
print(f"\nvideos={n}  acc={(tp + tn) / max(1, n):.2f}  TP={tp} FP={fp} FN={fn} TN={tn}")
if tc_err:
    print(f"type match on TP: {type_ok}/{tp}   timecode |err| median={sorted(tc_err)[len(tc_err) // 2]:.1f}s  "
          f"<=2s: {sum(e <= 2 for e in tc_err)}/{len(tc_err)}")
