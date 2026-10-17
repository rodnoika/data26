#!/usr/bin/env python3
"""Run detect_events.py over the labelled videos and score it.

Usage:
    python evaluate.py --videos open/videos --labels open/open_labels.xlsx
    python evaluate.py --videos open/videos --labels open/open_labels.xlsx --model qwen3-vl:32b --only X-1 X-2

Results are cached in --results (one JSON per video), so a rerun only scores.
Delete the folder or pass --rerun after changing prompts or settings.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import openpyxl

import detect_events

TOLERANCE_SEC = 3  # a timecode within this many seconds of the label counts as a hit


def norm(name):
    # The labels and file names mix Latin "X" and Cyrillic "Х".
    return name.strip().replace("Х", "X").replace("х", "x")


def parse_seconds(text):
    if not text:
        return None
    text = str(text)
    m = re.search(r"(\d+)\s*мин", text)
    s = re.search(r"(\d+)\s*сек", text)
    return (int(m.group(1)) * 60 if m else 0) + (int(s.group(1)) if s else 0)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path, help="Path to a single video file (e.g. X-1.mov)")
    ap.add_argument("--results", type=Path, default=Path("eval_results"))
    ap.add_argument("--rerun", action="store_true")
    # same model settings as detect_events.py
    for a in detect_events.build_parser()._actions:
        if a.dest not in ("help", "video", "out"):
            ap._add_action(a)
    args = ap.parse_args()

    video_path = args.video
    if not video_path.exists():
        sys.exit(f"Error: video file '{video_path}' not found.")

    name = norm(video_path.stem)
    args.results.mkdir(parents=True, exist_ok=True)
    cache = args.results / f"{name}.json"

    if cache.exists() and not args.rerun:
        pred = json.loads(cache.read_text(encoding="utf-8"))
    else:
        client = detect_events.connect(args.host, args.model)
        print(f"== {name}", file=sys.stderr)
        t0 = time.time()
        result, debug = detect_events.analyze(video_path, args, client)
        for v in result["violations"]:
            v.pop("evidence_frames", None)
        pred = {
            **result,
            "seconds": round(time.time() - t0, 1),
            "timeline": debug["timeline"],
            "roi": debug["roi"],
        }
        cache.write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(pred, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
