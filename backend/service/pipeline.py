"""analyze(path) -> API result: verdict + violations with evidence frames and normative references."""
import base64, os
import cv2

from service.signals import extract, read_frame
from service.rules import evaluate, review_hints, first_entry
from service import vlm, idcard

# "verdict": every safety-net hint counts as a possible violation (fewer misses, more false alarms).
# default: only hints that never fired on clean training videos count; the rest go to review_hints.
SAFETY_NET = os.environ.get("SAFETY_NET", "default")
from service.rules_catalog import CATALOG


def _evidence(path, t, focus, duration, extra_t=None):
    out = []
    times = [t, min(duration - 0.1, t + 1.0)] + ([extra_t] if extra_t and extra_t > t + 1.0 else [])
    for tt in times[:3]:
        fr = read_frame(path, max(0.0, tt))
        if fr is None:
            continue
        for b in focus:
            if b:
                x0, y0, x1, y1 = map(int, b)
                cv2.rectangle(fr, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(fr, f"{tt:.1f}s", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            out.append({"timestamp_sec": round(tt, 2), "image_base64": base64.b64encode(buf).decode()})
    return out


def vlm_checks(path, sig, events):
    """Targeted VLM questions on moments the rules select. Returns extra events + raw answers (for the cache)."""
    out, log = [], {}
    have = {e["type_id"] for e in events}
    t_e = first_entry(sig)
    if t_e is not None and "V11" not in have:
        closed, ans = vlm.door_closed(path, t_e)
        log["door"] = dict(entry=t_e, closed=closed, answer=ans)
        if not closed:
            out.append(dict(type_id="V11", t=round(t_e + 1.0, 2), focus=[], extra_t=round(t_e + 6.0, 2),
                            desc="Кассир входит в кассовое помещение и не закрывает за собой дверь "
                                 f"(проверка локальной VLM-моделью: «{ans}»)."))
    return out, log


def analyze(path, progress=None, signals=None, vlm_cache=None):
    """vlm_cache: precomputed vlm_checks() output (offline evaluation); None = run the VLM if enabled."""
    sig = signals or extract(path, progress)
    if signals is None and idcard.available():
        idcard.annotate(path, sig)                    # fine-tuned id/money detector (rule V6)
    def item(e, note=""):
        cat = CATALOG[e["type_id"]]
        return {
            "type": cat["type"],
            "timecode_sec": e["t"],
            "description": f"{e['desc']}{note} Основание: {cat['rule']}.",
            "rule": cat["rule"],
            "type_id": e["type_id"],
            "evidence_frames": _evidence(path, e["t"], e["focus"], sig["duration"], e.get("extra_t")),
        }

    events = evaluate(sig)
    if vlm_cache is not None:
        events = events + vlm_cache
    elif vlm.ENABLED:
        events = events + vlm_checks(path, sig, events)[0]
    events.sort(key=lambda e: e["t"])
    violations = [item(e) for e in events]
    review = []
    for h in review_hints(sig, events):
        if h.get("promote") or SAFETY_NET == "verdict":
            violations.append(item(h, " Признак выявлен страховочным правилом и требует проверки человеком."))
        else:
            review.append(item(h))
    violations.sort(key=lambda v: v["timecode_sec"])
    if progress:
        progress(1.0)
    return {"verdict": "violation" if violations else "no_violation", "violations": violations,
            "review_hints": review}
