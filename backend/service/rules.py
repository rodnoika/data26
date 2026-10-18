"""Rules engine: raw per-frame signals -> violation events (type id, timecode, description, focus boxes).

Pure python, no models, so it can be tuned against cached signals of the training set.
"""
from statistics import median

P = dict(
    person_score=0.5, person_min_h=0.07, person_min_w=0.10, desk_margin=0.01,  # box must end below the desk top line
    seat_above=0.08,                       # seated: box centre no higher than this above the desk top
    away_min=3.0, end_win=3.0, screen_unlocked=0.2, screen_white=0.09,
    smooth=3,                              # median window (samples) for people count
    thr=dict(money=0.30, bag=0.30, personal=0.28, docs=0.22, drink=0.25),
    item_min_frac=0.5,                     # share of frames in the window with the item
    flow_score=0.2, flow_lo=0.45, flow_hi=0.65, reach_win=6.0, reach_frac=0.3,
    hint_lone_cash=10.0,                   # s of cash in hands of a cashier who is not at the desk
    det_loc=0.30, bout_gap=3.0, det_lone_min=8.0, det_cash_min=3.0,
    id_score=0.5, id_sess_min=6.0, id_cash_min=1.0, id_ratio=0.4,
    key_score=0.6, key_gone_min=1.0,
    reentry_win=25.0, leave_win=10.0, count_min=8.0, handover_win=6.0,                      # s between a client leaving and the next one entering
    crowd_min=2.0,                         # s with >=3 people
    visit_min=3.0, visit_max=12.0,         # collector visit shorter than this = no document check
    stable=1.5,                            # s a people-count change must hold
    personal_min=3.0, bag_start=3.0, drink_min=2, drink_score=0.4, drink_raise=-0.02, empty_min=1.5,
    safe_open=0.85, safe_seated_min=3.0, safe_long=9.0, safe_loc=0.26,   # safe open while the cashier keeps working at the desk
    track_iou=0.25, track_gap=1.5,         # simple person tracker
    out_min=2.5, out_static=0.05, out_min_h=0.16, out_seat_frac=0.5, at_desk=0.06, sit_win=4.0, out_min_w=0.14,
)


_CTX = {}


def _people(fr, H):
    """People inside the cash room. Those seen through the glass walls stand behind the desk line:
    their box ends above the desk's top edge."""
    top = _CTX.get("desk_top", 0.0)
    return [b for n, s, b in fr["coco"]
            if n == "person" and s >= P["person_score"] and (b[3] - b[1]) >= P["person_min_h"] * H
            and b[3] >= top - P["desk_margin"] * H and (b[2] - b[0]) >= P["person_min_w"] * _CTX.get("W", 1e9)]


def _seated(fr, H):
    """People sitting at the desk: box centre at most a bit above the desk's top edge."""
    d = _CTX.get("desk", [0, 0, 1e9, 0])
    return [b for b in _people(fr, H)
            if (b[1] + b[3]) / 2 >= d[1] - P["seat_above"] * H and d[0] <= (b[0] + b[2]) / 2 <= d[2]]


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-6)


def _tracks(frames, H, W):
    """Greedy IoU / centre-distance tracker over confident person boxes (incl. people in the doorway)."""
    tracks = []
    for f in frames:
        dets = [b for n, s, b in f["coco"] if n == "person" and s >= P["person_score"] and b[3] - b[1] >= 0.12 * H]
        used = set()
        for tr in sorted(tracks, key=lambda tr: -tr["t"][-1]):
            if f["t"] - tr["t"][-1] > P["track_gap"]:
                continue
            last = tr["b"][-1]; lc = ((last[0] + last[2]) / 2, (last[1] + last[3]) / 2)
            best, bi = None, None
            for i, b in enumerate(dets):
                if i in used:
                    continue
                d = abs((b[0] + b[2]) / 2 - lc[0]) + abs((b[1] + b[3]) / 2 - lc[1])
                score = _iou(last, b) - d / W
                if (_iou(last, b) >= P["track_iou"] or d < 0.12 * W) and (best is None or score > best):
                    best, bi = score, i
            if bi is not None:
                used.add(bi); tr["t"].append(f["t"]); tr["b"].append(dets[bi])
        for i, b in enumerate(dets):
            if i not in used:
                tracks.append(dict(t=[f["t"]], b=[b]))
    return tracks


def _smooth(xs, k):
    h = k // 2
    return [int(median(xs[max(0, i - h):i + h + 1])) for i in range(len(xs))]


def _spans(flags, ts, min_len):
    """[(t_start, t_end, i_start, i_end)] of consecutive True runs lasting >= min_len seconds."""
    out, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            if ts[j] - ts[i] + (ts[1] - ts[0] if len(ts) > 1 else 0) >= min_len:
                out.append((ts[i], ts[j], i, j))
            i = j + 1
        else:
            i += 1
    return out


def _item_score(fr, group):
    return max([s for g, s, b in fr["desk"] if g == group] + [0.0])


def _item_box(fr, group):
    c = [(s, b) for g, s, b in fr["desk"] if g == group]
    return max(c)[1] if c else None


def _present(frames, i0, i1, group):
    seg = frames[i0:i1 + 1]
    if not seg:
        return False
    hits = sum(_item_score(f, group) >= P["thr"][group] for f in seg)
    return hits / len(seg) >= P["item_min_frac"]


def _seen(frames, i0, i1, group, k=3):
    """The item is seen in at least k frames of the window (robust to occlusion by people walking by)."""
    return sum(_item_score(f, group) >= P["thr"][group] for f in frames[i0:i1 + 1]) >= k


def evaluate(sig):
    fr, H = sig["frames"], sig["height"]
    if not fr:
        return []
    _CTX["desk"] = sig.get("layout", {}).get("desk", {}).get("box", [0, 0, 1e9, 0])
    _CTX["desk_top"] = _CTX["desk"][1]; _CTX["W"] = sig["width"]
    ts = [f["t"] for f in fr]
    dt = ts[1] - ts[0] if len(ts) > 1 else 0.5
    win = lambda t0, t1: (next((i for i, t in enumerate(ts) if t >= t0), len(ts) - 1),
                          max([i for i, t in enumerate(ts) if t <= t1] + [0]))
    cnt = _smooth([len(_people(f, H)) for f in fr], P["smooth"])
    ev = []

    def add(vid, t, desc, focus=None, extra_t=None):
        ev.append(dict(type_id=vid, t=round(max(0.0, t), 2), desc=desc, focus=focus or [],
                       extra_t=round(extra_t, 2) if extra_t is not None else None))

    # --- cashier left the workplace: nobody seated at the desk ---
    seated = _smooth([int(bool(_seated(f, H))) for f in fr], P["smooth"])
    for t0, t1, i0, i1 in _spans([c == 0 for c in seated], ts, P["away_min"]):
        if t0 <= ts[0] + 1e-6:          # nobody seated from the very start: nobody "left"
            continue
        empty = [cnt[k] == 0 for k in range(i0, i1 + 1)]   # standing at the desk / safe is not "gone"
        if not (_spans(empty, ts[i0:i1 + 1], P["empty_min"]) or (i1 == len(ts) - 1 and empty[-1])):
            continue
        # what is left behind is judged at the end of the away period (cash may be carried to the safe)
        j1 = i1; j0 = max(i0, i1 - int(P["end_win"] / dt))
        if _present(fr, j0, j1, "money") or _present(fr, j0, j1, "bag"):
            g = "money" if _present(fr, j0, j1, "money") else "bag"
            add("V1", t0, "Кассир покидает рабочее место, на столе остаются "
                + ("наличные денежные средства" if g == "money" else "сумка с ценностями")
                + " без присмотра.", [_item_box(fr[j1], g)])
        elif _present(fr, j0, j1, "docs") or _present(fr, j0, j1, "drink"):
            add("V4", t0, "Кассир покидает рабочее место, на столе остаются документы / предметы, "
                "не убранные на хранение.", [_item_box(fr[j1], "docs") or _item_box(fr[j1], "drink")])
        scr = [f["screen_p"][0] for f in fr[j0:j1 + 1] if f.get("screen_p")]
        # bright white app content: SigLIP reads a darkened room (light switched off, Х-11) as "screen off"
        white = [f["screen"] for f in fr[i0:i1 + 1] if f.get("screen") is not None]
        if (scr and median(scr) >= P["screen_unlocked"]) or (white and median(white) >= P["screen_white"]):
            t5 = t0 if scr and median(scr) >= P["screen_unlocked"] else next(
                (f["t"] for f in fr[i0:i1 + 1] if (f.get("screen") or 0) >= P["screen_white"]), t0)   # lights off
            add("V5", t5, "Кассир покидает рабочее место, экран рабочей станции остаётся разблокированным "
                "(видно открытое приложение).", [fr[j1]["laptop"]])

    # --- cashier left the room with the key still in the safe lock (fine-tuned key detector) ---
    if any("key" in f for f in fr):
        for t0, t1, i0, i1 in _spans([c == 0 for c in seated], ts, P["away_min"]):
            if t0 <= ts[0] + 1e-6 or not any(cnt[k] == 0 for k in range(i0, i1 + 1)):
                continue
            gone = [k for k in range(i0, i1 + 1) if cnt[k] == 0]          # nobody in the room at all
            ks = [fr[k]["key"][0] if fr[k].get("key") else 0.0 for k in gone]
            if len(gone) * dt >= P["key_gone_min"] and median(ks) >= P["key_score"]:
                k0 = gone[0]
                add("V2", ts[k0], "Кассир покидает помещение, ключ остаётся в замке сейфа "
                    "(сейф не заперт / ключ не изъят).", [fr[k0]["key"][1] if fr[k0].get("key") else None],
                    extra_t=min(ts[-1], ts[k0] + 3))
                break

    # --- safe left open ---
    safe_ok = sig["layout"].get("safe", {}).get("score", 0) >= P["safe_loc"]   # unsure where the safe is: skip
    so = [(f.get("safe_open") or 0.0) if safe_ok else 0.0 for f in fr]
    so_s = [median(so[max(0, i - 1):i + 2]) for i in range(len(so))]
    sb = sig["layout"].get("safe", {}).get("box")
    # normal use = open, take/put cash, close (<= ~7.5 s in the training set). Violation: it stays open.
    for t0, t1, i0, i1 in _spans([v >= P["safe_open"] for v in so_s], ts, P["safe_seated_min"]):
        dur = t1 - t0 + dt
        seated_open = sum(seated[i] > 0 for i in range(i0, i1 + 1)) * dt
        if dur >= P["safe_long"] or (i1 == len(ts) - 1 and seated_open >= P["safe_seated_min"]):
            ts_ = next((ts[i] for i in range(i0, i1 + 1) if seated[i] > 0), t0)   # sat down, safe still open
            add("V2", ts_, f"Сейф остаётся открытым ≈{dur:.0f} с и не закрывается после выемки / закладки ценностей.",
                [sb], extra_t=min(t1, ts_ + 4))
            break
    for t0, t1, i0, i1 in _spans([c == 0 for c in seated], ts, P["away_min"]):
        if t0 > ts[0] + 1e-6 and any(cnt[k] == 0 for k in range(i0, i1 + 1)):
            j0 = max(i0, i1 - int(P["end_win"] / dt))
            if median(so_s[j0:i1 + 1]) >= P["safe_open"]:
                add("V2", t0, "Кассир покидает рабочее место, сейф остаётся открытым.",
                    [sig["layout"].get("safe", {}).get("box")])

    # --- outsider: a newcomer who stays, never sits and never comes to the desk ---
    W = sig["width"]; top = _CTX["desk_top"]
    for tr in _tracks(fr, H, W):
        if tr["t"][0] <= ts[0] + 1e-6 or tr["t"][-1] - tr["t"][0] < P["out_min"]:
            continue
        bs = tr["b"]
        if median(b[3] - b[1] for b in bs) < P["out_min_h"] * H or median(b[2] - b[0] for b in bs) < P["out_min_w"] * W:
            continue                                                 # small / narrow: seen through the glass
        if any(b[3] >= top + P["at_desk"] * H for b in bs):          # came to the desk: client / collector
            continue
        if sum(bool(_seated({"coco": [("person", 1.0, b)]}, H)) for b in bs) > P["out_seat_frac"] * len(bs):
            continue
        tail = [b for t, b in zip(tr["t"], bs) if t >= tr["t"][-1] - P["out_min"]]
        cx = [(b[0] + b[2]) / 2 for b in tail]
        if max(cx) - min(cx) > P["out_static"] * W * 2:              # walking past behind the glass
            continue
        i0 = next(i for i, t in enumerate(ts) if t >= tr["t"][0])
        if not any(seated[i] > 0 for i in range(i0, min(len(ts), i0 + int(2 / dt) + 1))):
            continue                                                 # nobody working at the desk
        n_seat = [len(_seated(f, H)) for f in fr]
        later = [n_seat[i] for i, t in enumerate(ts) if tr["t"][0] <= t <= tr["t"][-1] + P["sit_win"]]
        if later and max(later) > max(1, n_seat[i0]):                 # someone sat down: a client
            continue
        add("V11", tr["t"][0], "В кассовое помещение входит / стоит в дверях постороннее лицо, которое не "
            "обслуживается за столом; доступ в кассовое помещение не ограничен.", [bs[0]])
        break

    # --- client leaves with cash on the desk; next person enters before it is stored ---
    stab = int(P["stable"] / dt)
    for i in range(stab, len(cnt) - stab):
        if all(c >= 2 for c in cnt[i - stab:i]) and all(c == 1 for c in cnt[i:i + stab]):
            t_exit = ts[i]
            a, b = win(t_exit - 4, t_exit + 3)            # cash may be hidden while the client walks out
            if not _seen(fr, a, b, "money"):
                continue
            k = next((j for j in range(i + 1, len(cnt) - stab + 1) if all(c >= 2 for c in cnt[j:j + stab])
                      and ts[j] - t_exit <= P["reentry_win"]), None)
            if k is not None:
                a2, b2 = win(ts[k] - 1, ts[k] + 3)
                if _seen(fr, a2, b2, "money"):
                    add("V3", ts[k], "Предыдущий клиент ушёл, наличные остаются на столе; в помещение входит "
                        "следующее лицо, ценности не убраны в хранилище.", [_item_box(fr[min(len(fr) - 1, k + 2)], "money")])

    # --- client stands up and leaves right after handing over cash: counted without the client ---
    nseat = _smooth([len(_seated(f, H)) for f in fr], P["smooth"])
    for i in range(stab, len(nseat) - stab):
        if not (all(c >= 2 for c in nseat[i - stab:i]) and all(c <= 1 for c in nseat[i:i + stab])):
            continue
        t_up = ts[i]
        j0 = i - 1
        while j0 > 0 and nseat[j0 - 1] >= 2:             # start of this client's seated session
            j0 -= 1
        m = next((k for k in range(j0, i) if _item_score(fr[k], "money") >= P["thr"]["money"]), None)
        gone = next((k for k in range(i, len(cnt)) if cnt[k] < cnt[i - 1] and ts[k] - t_up <= P["leave_win"]), None)
        if (m is not None and gone is not None and t_up - ts[m] <= P["count_min"]
                and ts[m] - ts[j0] <= P["handover_win"]):          # cash handed over right after sitting down
            add("V7", t_up, "Клиент передаёт наличные и покидает кассу через "
                f"{t_up - ts[m]:.0f} с, пересчёт выполняется не в его присутствии.", [_item_box(fr[m], "money")])
            break

    # --- banknote taken from the counter and handed back to the client (suspicious note returned) ---
    cb = sig["layout"].get("counter", {}).get("box")
    if cb:
        def cash_boxes(f):   # zero-shot OWLv2 desk cash + the fine-tuned money detector (robust to mirroring)
            out = [b for g, sc, b in f["desk"] if g == "money" and sc >= P["flow_score"]]
            if f.get("money") and f["money"][0] >= P["id_score"]:
                out.append(f["money"][1])
            return out
        money_boxes = [b for f in fr for b in cash_boxes(f)]
        static = lambda b: sum(_iou(b, o) > 0.7 for o in money_boxes) >= 6      # ID card lying on the desk
        e = [cb[0] - 10, cb[1] - 10, cb[2] + 10, cb[3] + 10]; ea = (e[2] - e[0]) * (e[3] - e[1])
        reach, flow, n2 = [], [], 0
        for f in fr:
            st = _seated(f, H)
            if len(st) < 2:
                continue
            n2 += 1
            cash_b, cli_b = max(st, key=lambda b: b[3]), min(st, key=lambda b: b[3])   # cashier: nearer box
            ix = max(0, min(cash_b[2], e[2]) - max(cash_b[0], e[0])) * max(0, min(cash_b[3], e[3]) - max(cash_b[1], e[1]))
            if ix > 0.05 * ea:
                reach.append(f["t"])
            c0 = ((cash_b[0] + cash_b[2]) / 2, (cash_b[1] + cash_b[3]) / 2)
            c1 = ((cli_b[0] + cli_b[2]) / 2, (cli_b[1] + cli_b[3]) / 2)
            vx, vy = c1[0] - c0[0], c1[1] - c0[1]; L2 = vx * vx + vy * vy or 1.0
            for b in cash_boxes(f):
                if not static(b):
                    m = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                    flow.append((f["t"], ((m[0] - c0[0]) * vx + (m[1] - c0[1]) * vy) / L2, b))
        if n2 and len(reach) <= P["reach_frac"] * n2:     # cashier's box touching the counter is unusual here
            hits = []
            for i, (t0, s0, b0) in enumerate(flow):
                if s0 > P["flow_lo"] or not any(0 <= t0 - r <= P["reach_win"] for r in reach):
                    continue                                  # cash not at the cashier, or no reach just before
                up = next(((t1, b1) for t1, s1, b1 in flow[i + 1:] if 0 < t1 - t0 <= 3 and s1 >= P["flow_hi"]), None)
                if up:
                    hits.append((t0, b0, up))
            if hits:                                          # the return comes after the inspection: last one
                t0, b0, up = hits[-1]
                add("V8", t0, "Кассир достаёт банкноты из счётчика (лоток отбраковки), осматривает их и "
                    "возвращает клиенту; порядок работы с сомнительными банкнотами (акт, изъятие) не соблюдён.",
                    [b0, up[1]], extra_t=up[0])

    # --- counter used by a lone cashier, rejected notes never checked on the detector device ---
    dv, cb = sig["layout"].get("detector"), sig["layout"].get("counter", {}).get("box")
    if dv and cb and dv["score"] >= P["det_loc"] and _iou(dv["box"], cb) < 0.5:   # same box = localisation failed
        db = dv["box"]; e = [cb[0] - 10, cb[1] - 10, cb[2] + 10, cb[3] + 10]
        reach_t, on_det, lone = [], 0, 0
        for i, f in enumerate(fr):
            st = _seated(f, H)
            if len(st) != 1 or cnt[i] != 1:
                continue
            lone += 1
            c = st[0]
            ix = max(0, min(c[2], e[2]) - max(c[0], e[0])) * max(0, min(c[3], e[3]) - max(c[1], e[1]))
            if ix > 0.05 * (e[2] - e[0]) * (e[3] - e[1]):
                reach_t.append(f["t"])
            on_det += any(g == "money" and sc >= P["flow_score"] and _iou(b, db) > 0 for g, sc, b in f["desk"])
        bouts = [t for k, t in enumerate(reach_t) if k == 0 or t - reach_t[k - 1] > P["bout_gap"]]
        if lone * dt >= P["det_lone_min"] and len(bouts) >= 2 and on_det == 0                 and sum(_item_score(f, "money") >= P["thr"]["money"] for f in fr) * dt >= P["det_cash_min"]:
            t_hand = min(ts[-1], reach_t[-1] + 2.0)       # notes out of the counter, counted by hand
            add("V8", t_hand, "Кассир извлекает банкноты из счётчика после пересчёта и пересчитывает их вручную, "
                "не проверив на детекторе подлинности банкнот (отбракованные банкноты не проверены).",
                [cb, db], extra_t=min(ts[-1], t_hand + 3))

    # --- cash accepted without the client showing an ID document (fine-tuned id/money detector) ---
    if any("id" in f for f in fr):
        i = 0
        while i < len(nseat):
            if nseat[i] < 2:
                i += 1; continue
            j = i
            while j + 1 < len(nseat) and nseat[j + 1] >= 2:
                j += 1
            a = max(0, i - int(3 / dt))                       # the card is often shown while sitting down
            win = fr[a:j + 1]
            id_s = sum(1 for f in win if f.get("id") and f["id"][0] >= P["id_score"]) * dt
            cash = [f for f in win if (f.get("money") and f["money"][0] >= P["id_score"])
                    or _item_score(f, "money") >= P["thr"]["money"]]
            sess = (j - i + 1) * dt
            if sess >= P["id_sess_min"] and len(cash) * dt >= P["id_cash_min"] and id_s / sess < P["id_ratio"]:
                t_cash = cash[0]["t"]
                add("V6", t_cash, "Клиент передаёт наличные кассиру, документ, удостоверяющий личность, не "
                    f"предъявляется (удостоверение видно ≈{id_s:.0f} с из {sess:.0f} с обслуживания).",
                    [(cash[0].get("money") or [0, _item_box(cash[0], "money")])[1]])
                break
            i = j + 1

    # --- third person during service ---
    for t0, t1, i0, i1 in _spans([c >= 3 for c in cnt], ts, P["crowd_min"]):
        add("V12", t0, "Во время обслуживания клиента в зоне кассы находится третье лицо.", _people(fr[i0], H))
        break

    # --- personal belongings on the desk ---
    # generic COCO "handbag": only a bag that ARRIVES (not there at the start) and is not desk equipment;
    # on unusual angles the counting machine / laptop are read as bags (robustness test, warp variant)
    fixed = [b for f in fr if f["t"] <= P["bag_start"] for n, sc, b in f["coco"] if n in ("handbag", "backpack") and sc >= .3]
    fixed += [b for f in fr if f["t"] <= P["bag_start"] for g, sc, b in f["desk"] if g == "personal" and sc >= .2]
    fixed += [sig["layout"][k]["box"] for k in ("counter", "laptop") if sig["layout"].get(k)]
    def new_bag(f):
        return any(n in ("handbag", "backpack") and sc >= 0.5 and not any(_iou(b, o) > 0.3 for o in fixed)
                   for n, sc, b in f["coco"])
    def new_item(f):
        return any(g == "personal" and sc >= P["thr"]["personal"] and not any(_iou(b, o) > 0.3 for o in fixed)
                   for g, sc, b in f["desk"])
    pers = [new_item(f) or new_bag(f) for f in fr]
    for t0, t1, i0, i1 in _spans(pers, ts, P["personal_min"]):
        add("V13", t0, "На рабочем месте кассира находятся личные вещи (сумка / верхняя одежда).",
            [_item_box(fr[i0], "personal")])
        break

    # --- drinking: cup/bottle at the face of a person ---
    run = 0
    for i, f in enumerate(fr):
        hit = None
        for n, s, b in f["coco"]:
            if n in ("cup", "bottle", "wine glass") and s >= P["drink_score"]:
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if cy >= _CTX["desk_top"] - P["drink_raise"] * H:     # not raised to the mouth: cash / hands
                    continue
                for p in _people(f, H):
                    if p[0] <= cx <= p[2] and p[1] <= cy <= p[1] + 0.35 * (p[3] - p[1]):
                        hit = b
        run = run + 1 if hit else 0
        if run >= P["drink_min"]:
            add("V14", ts[i - run + 1], "Кассир употребляет напиток на рабочем месте рядом с наличными.", [hit])
            break

    # --- short collector visit with a cash bag ---
    for i in range(max(1, stab), len(cnt) - stab):
        if ts[i] < 1.0 or not (all(c > cnt[i - 1] for c in cnt[i:i + stab]) and cnt[i - 1] >= 1):
            continue
        j = next((j for j in range(i + stab, len(cnt) - stab + 1) if all(c <= cnt[i - 1] for c in cnt[j:j + stab])), None)
        if j is None or not (P["visit_min"] <= ts[j] - ts[i] <= P["visit_max"]):
            continue
        bag_i = next((k for k in range(i, j + 1) if _item_score(fr[k], "bag") >= P["thr"]["bag"]), None)
        if bag_i is not None:
            add("V10", ts[bag_i], "Инкассатор передаёт сумку и покидает помещение через несколько секунд; "
                "проверка сопроводительных документов и пломбы не видна.", [_item_box(fr[bag_i], "bag")])
            break

    # one event per type (earliest)
    first = {}
    for e in sorted(ev, key=lambda e: e["t"]):
        first.setdefault(e["type_id"], e)
    return sorted(first.values(), key=lambda e: e["t"])


def review_hints(sig, events):
    """Safety net: moments the rules cannot judge but a human should look at (unseen violation types).
    Hints with promote=True never fired on a clean training video and count toward the verdict; the rest
    only count when SAFETY_NET=verdict (see pipeline)."""
    fr, H = sig["frames"], sig["height"]
    if not fr:
        return []
    evaluate(sig)                                        # sets the per-video context (desk, width)
    ts = [f["t"] for f in fr]
    dt = ts[1] - ts[0] if len(ts) > 1 else 0.5
    cnt = _smooth([len(_people(f, H)) for f in fr], P["smooth"])
    seated = _smooth([int(bool(_seated(f, H))) for f in fr], P["smooth"])
    have = {e["type_id"] for e in events}
    hints = []

    # R1: cashier leaves the room; things not visible to the rules (key in the safe, door, cash in pockets)
    if not have & {"V1", "V2", "V4", "V5"}:
        for t0, t1, i0, i1 in _spans([c == 0 for c in seated], ts, P["away_min"]):
            if t0 > ts[0] + 1e-6 and _spans([cnt[k] == 0 for k in range(i0, i1 + 1)], ts[i0:i1 + 1], P["empty_min"]) \
                    or (t0 > ts[0] + 1e-6 and i1 == len(ts) - 1 and cnt[i1] == 0):
                hints.append(dict(type_id="R1", t=t0, focus=[sig["layout"].get("safe", {}).get("box")],
                                  desc="Кассир покидает рабочее место / помещение. Проверьте вручную: сейф заперт и "
                                       "ключ извлечён, дверь кассового помещения закрыта, ценности убраны."))
                break

    # R2: cash handled by a lone cashier who is not seated at the desk
    lone = [cnt[i] == 1 and seated[i] == 0 and _item_score(f, "money") >= P["thr"]["money"] for i, f in enumerate(fr)]
    if sum(lone) * dt >= P["hint_lone_cash"]:
        i = len(lone) - 1 - lone[::-1].index(True)          # last moment the cash is seen in the cashier's hands
        t = min(ts[-1], ts[i] + 1.0)
        hints.append(dict(type_id="R2", t=t, focus=[_item_box(fr[i], "money")], promote=True,
                          extra_t=min(ts[-1], t + 3.0),
                          desc="Кассир работает с наличными стоя, вне рабочего места; затем наличные пропадают из "
                               "поля зрения не на столе. Проверьте, куда перемещаются денежные средства "
                               "(хранилище / сейф / одежда)."))

    return sorted(hints, key=lambda h: h["t"])


def first_entry(sig):
    """Time the first person walks into an empty room (cashier opening the cash room), or None."""
    fr, H = sig["frames"], sig["height"]
    if len(fr) < 4:
        return None
    evaluate(sig)                                        # per-video context
    ts = [f["t"] for f in fr]
    dt = ts[1] - ts[0]
    cnt = _smooth([len(_people(f, H)) for f in fr], P["smooth"])
    stab = int(P["stable"] / dt)
    if cnt[0] != 0:
        return None
    for i in range(1, len(cnt) - stab):
        if all(c >= 1 for c in cnt[i:i + stab]):
            return ts[i]
    return None
