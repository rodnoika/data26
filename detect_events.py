#!/usr/bin/env python3
"""Find cash and ID-card events in cashier CCTV with Qwen3-VL (Ollama, Apache 2.0).

Why the plain description script misses banknotes and ID cards:
the videos are 464x848, the desk fills a quarter of the frame, and a banknote
is 15-25 px wide. Qwen3-VL turns every 32x32 px block into one token, so the
object covers less than one token and the model cannot see it.

What this script does instead:
1. Asks the model once per video where the cashier's desk is (works for any camera angle).
2. Crops the desk and upscales it 2x, so small objects span several tokens.
3. Samples 2 frames per second in 4-second windows, so a 1-second hand-over is not skipped.
4. Asks fixed questions with a JSON answer instead of a free description.
5. Turns the event timeline into a verdict with timecodes and evidence frames (the jury API schema).

Usage:
    python detect_events.py video.mov
    python detect_events.py video.mov --model qwen3-vl:32b --fps 3
"""
import argparse
import base64
import io
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

ROI_PROMPT = """This is a CCTV frame from a bank cash desk.
Find the work area where cash and documents change hands: the cashier's desk top,
including the counting machine and the hands of the cashier and the client.
Answer ONLY with JSON: {"bbox_2d": [x1, y1, x2, y2]} using coordinates normalized to 0-1000."""

WINDOW_PROMPT = """You are reviewing bank cashier CCTV. The camera looks down at one cash desk.
The cashier sits at the desk; clients and cash collectors come from the other side.

Images, in order:
{image_list}
Close-ups are enlarged crops of the desk. Use them to see small objects.

Look for:
- CASH: paper banknotes. From above they look like small light rectangles (beige, green, blue, pink),
  as a stack, a fan, a bundle in a hand, or in the counting machine.
- ID CARD: a plastic card the size of a bank card, often blue or white, held in a hand,
  lying on the desk, or being looked at by the cashier.
- Who holds each object and where it goes: hand-over between client and cashier, left on the desk,
  put into a pocket, bag, drawer, or safe.
- Cashier leaving the desk, safe door open, an extra person behind the desk, phone use, lights off.

Answer ONLY with JSON:
{{"cash_visible": true or false,
 "id_card_visible": true or false,
 "events": [{{"t": seconds, "actor": "cashier|client|collector|other",
             "action": "short phrase", "object": "cash|id_card|document|phone|safe|none"}}],
 "notes": "one sentence about anything unusual"}}
Report only what you see. Leave out objects you are unsure about."""

VERDICT_PROMPT = """You are an assistant to a bank cash-desk auditor.
Below is a timeline of events seen on a CCTV video ({duration:.0f} s long), followed by the rules to check.

TIMELINE:
{timeline}

RULES:
{rules}

Decide which rules the video shows signs of breaking. Use only events in the timeline.
Do not flag a rule if checking it needs data that a camera cannot show.
Write descriptions in {language}.

Answer ONLY with JSON:
{{"verdict": "violation" or "no_violation",
 "violations": [{{"rule_id": "R1", "timecode_sec": seconds,
                 "description": "what is visible and why it may break the rule"}}]}}"""

# Starting rule set. Edit names and conditions, and fill "source" from adilet.zan.kz (NBRK acts).
RULES = [
    {"id": "R1", "name": "Выдача наличных без проверки документа",
     "condition": "Cashier hands cash to a client and no ID card was shown to or checked by the cashier before it."},
    {"id": "R2", "name": "Наличные или документ оставлены без присмотра",
     "condition": "Cash or an ID card lies on the desk while the cashier is away from the desk or turned away."},
    {"id": "R3", "name": "Кассир покинул рабочее место",
     "condition": "Cashier leaves the desk or the room while cash is out or the safe is open."},
    {"id": "R4", "name": "Открытый сейф без присмотра",
     "condition": "Safe door stays open while nobody is next to it, or the room light goes off with the safe open."},
    {"id": "R5", "name": "Посторонний в зоне кассира",
     "condition": "A person other than the cashier is behind the desk or touches cash in the cashier's area."},
    {"id": "R6", "name": "Наличные убраны вне учёта",
     "condition": "Cashier puts cash into a pocket, bag or clothing instead of the drawer, safe, or client."},
    {"id": "R7", "name": "Пересчёт вне зоны видимости камеры",
     "condition": "Cash is counted or handled below the desk or hidden from the camera."},
    {"id": "R8", "name": "Использование телефона при работе с наличными",
     "condition": "Cashier uses a mobile phone while cash is on the desk or in hand."},
]


# ---------- frames ----------

def read_frames(video, fps, tmp):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"fps={fps}",
                    "-q:v", "2", str(tmp / "f_%05d.jpg")], check=True)
    paths = sorted(tmp.glob("f_*.jpg"))
    return [(i / fps, Image.open(p).convert("RGB")) for i, p in enumerate(paths)]


def to_jpeg(img, quality=90):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def crop_zoom(img, box, max_side):
    crop = img.crop(box)
    scale = max_side / max(crop.size)
    return crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)


# ---------- model calls ----------

def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:200]}")
    return json.loads(m.group(0))


def ask(client, model, prompt, images=(), num_ctx=16384, retries=2):
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = list(images)
    for attempt in range(retries + 1):
        resp = client.chat(model=model, messages=[msg], format="json",
                           options={"temperature": 0.1, "num_ctx": num_ctx})
        try:
            return parse_json(resp["message"]["content"])
        except (ValueError, json.JSONDecodeError):
            if attempt == retries:
                raise


def find_roi(client, model, frames, pad=0.12):
    """Union of desk boxes found on 3 frames, padded. Falls back to the full frame."""
    w, h = frames[0][1].size
    boxes = []
    for idx in {len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4}:
        try:
            b = ask(client, model, ROI_PROMPT, [to_jpeg(frames[idx][1])], num_ctx=4096)["bbox_2d"]
            x1, y1, x2, y2 = [float(v) for v in b]
            if max(x1, y1, x2, y2) <= 1000 and x2 > x1 and y2 > y1:
                boxes.append((x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h))
        except Exception as e:
            print(f"[roi] frame {idx}: {e}", file=sys.stderr)
    if not boxes:
        return (0, 0, w, h)
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    box = (max(0, int(x1 - px)), max(0, int(y1 - py)), min(w, int(x2 + px)), min(h, int(y2 + py)))
    # A box covering most of the frame gives no zoom; keep it anyway so the run continues.
    return box


# ---------- pipeline ----------

def analyze(video, args, client):
    t_start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        frames = read_frames(video, args.fps, Path(tmp))
    if not frames:
        sys.exit("ffmpeg returned no frames")
    duration = frames[-1][0] + 1 / args.fps

    roi = find_roi(client, args.model, frames)
    print(f"[roi] {roi} of {frames[0][1].size}", file=sys.stderr)

    per_window = max(1, round(args.window * args.fps))
    timeline = []
    for s in range(0, len(frames), per_window):
        chunk = frames[s:s + per_window]
        images, labels = [], []
        # one wide view at the start of the window for context (people, safe, lights)
        images.append(to_jpeg(chunk[0][1]))
        labels.append(f"Image 1: full view at t={chunk[0][0]:.1f}s")
        for t, img in chunk:
            images.append(to_jpeg(crop_zoom(img, roi, args.crop_size)))
            labels.append(f"Image {len(images)}: desk close-up at t={t:.1f}s")
        t0 = time.time()
        try:
            res = ask(client, args.model, WINDOW_PROMPT.format(image_list="\n".join(labels)), images)
        except Exception as e:
            print(f"[{chunk[0][0]:.1f}s] failed: {e}", file=sys.stderr)
            continue
        res["window"] = [chunk[0][0], chunk[-1][0]]
        timeline.append(res)
        print(f"[{chunk[0][0]:5.1f}-{chunk[-1][0]:5.1f}s] {time.time() - t0:4.1f}s "
              f"cash={res.get('cash_visible')} id={res.get('id_card_visible')} "
              f"events={len(res.get('events', []))}", file=sys.stderr)

    lines = []
    for w in timeline:
        for e in w.get("events", []):
            lines.append(f"{float(e.get('t', w['window'][0])):.1f}s {e.get('actor')}: {e.get('action')} [{e.get('object')}]")
        flags = [k for k in ("cash_visible", "id_card_visible") if w.get(k)]
        if w.get("notes") or flags:
            lines.append(f"{w['window'][0]:.1f}-{w['window'][1]:.1f}s visible={flags} note: {w.get('notes', '')}")
    rules_txt = "\n".join(f"{r['id']}: {r['name']} - {r['condition']}" for r in RULES)
    verdict = ask(client, args.model, VERDICT_PROMPT.format(
        duration=duration, timeline="\n".join(lines) or "(no events)", rules=rules_txt,
        language="Russian" if args.lang == "ru" else "English"), num_ctx=32768)

    rule_by_id = {r["id"]: r for r in RULES}
    violations = []
    for v in verdict.get("violations", []):
        t = float(v.get("timecode_sec", 0))
        nearest = min(frames, key=lambda f: abs(f[0] - t))
        rule = rule_by_id.get(v.get("rule_id"), {})
        violations.append({
            "type": rule.get("name", v.get("rule_id", "unknown")),
            "timecode_sec": round(nearest[0], 1),
            "description": v.get("description", ""),
            "rule_id": v.get("rule_id"),
            "evidence_frames": [
                {"timestamp_sec": round(nearest[0], 1),
                 "image_base64": base64.b64encode(to_jpeg(nearest[1], 85)).decode()},
                {"timestamp_sec": round(nearest[0], 1),
                 "image_base64": base64.b64encode(to_jpeg(crop_zoom(nearest[1], roi, args.crop_size), 85)).decode()},
            ],
        })
    result = {"status": "done", "verdict": "violation" if violations else "no_violation",
              "violations": violations}
    debug = {"roi": roi, "timeline": timeline, "seconds": round(time.time() - t_start, 1)}
    return result, debug


def build_parser():
    ap = argparse.ArgumentParser(description="Detect cash-desk violations in a video.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--fps", type=float, default=2, help="frames per second to check (default: 2)")
    ap.add_argument("--window", type=float, default=4, help="seconds per model call (default: 4)")
    ap.add_argument("--crop-size", type=int, default=896, help="long side of the desk close-up in px")
    ap.add_argument("--lang", default="ru", choices=["ru", "en"])
    ap.add_argument("--out", type=Path, default=None, help="write result JSON here")
    return ap


def connect(host, model):
    import ollama
    client = ollama.Client(host=host)
    try:
        client.show(model)
    except ollama.ResponseError:
        sys.exit(f"Model {model} is not pulled. Run: ollama pull {model}")
    except Exception as e:
        sys.exit(f"Cannot reach Ollama at {host}: {e}")
    return client


def main():
    args = build_parser().parse_args()
    client = connect(args.host, args.model)
    result, debug = analyze(args.video.resolve(), args, client)
    out = args.out or args.video.with_suffix(".result.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".debug.json").write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**result, "violations": [{k: v for k, v in x.items() if k != "evidence_frames"}
                                               for x in result["violations"]]}, ensure_ascii=False, indent=2))
    print(f"\n{debug['seconds']}s. Saved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
