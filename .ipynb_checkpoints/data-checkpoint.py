#!/usr/bin/env python3
"""Describe a video with local models that allow commercial use.

Vision + text: Qwen3-VL served by Ollama (Apache 2.0)
Speech:        faster-whisper (MIT) with Whisper weights (MIT)

Usage:
    python describe_video.py input.mp4
    python describe_video.py input.mp4 --model qwen3-vl:30b --segment 20 --frames 6
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SEGMENT_PROMPT = """You are a fraud detection system that is given a camera footage of one of national bank's cabins and you should output a description of every action in detail for further analysis. Especially focus on any suspicious behaviour as it can be a form of fraud"""

SUMMARY_PROMPT = """Below are timestamped notes about consecutive parts of one video.

Write, in {language}:
1. TITLE: one line.
2. DESCRIPTION: 
Notes:
{notes}"""


def fmt_time(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def probe_duration(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def extract_frames(video, start, length, count, out_dir, width):
    """Grab `count` evenly spaced frames from [start, start+length)."""
    frames = []
    for i in range(count):
        t = start + length * (i + 0.5) / count
        path = out_dir / f"f_{start:08.1f}_{i}.jpg"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video),
             "-frames:v", "1", "-vf", f"scale='min({width},iw)':-2", "-q:v", "3", str(path)],
            check=True,
        )
        if path.exists():
            frames.append(path)
    return frames

def load_audio(video):
    """Decode audio to 16 kHz mono float32 with ffmpeg. Returns None if there is no audio track."""
    import numpy as np

    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
    )
    if out.returncode != 0 or not out.stdout:
        return None
    return np.frombuffer(out.stdout, np.float32)
def transcribe(video, model_size, device, language):
    from faster_whisper import WhisperModel
    audio = load_audio(video)
    if audio is None:
        print("[whisper] no audio track, skipping speech", file=sys.stderr)
        return [], None

    if device == "auto":
        import ctranslate2
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[whisper] loading {model_size} on {device} ({compute_type})", file=sys.stderr)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(video), vad_filter=True, language=language)
    lines = [(s.start, s.end, s.text.strip()) for s in segments]
    print(f"[whisper] language={info.language}, {len(lines)} lines", file=sys.stderr)
    return lines, info.language


def transcript_for(lines, start, end):
    return " ".join(text for s, e, text in lines if e > start and s < end)


def chat(client, model, prompt, images=None, num_ctx=16384):
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = [str(p) for p in images]
    resp = client.chat(model=model, messages=[msg], options={"temperature": 0.2, "num_ctx": num_ctx})
    return resp["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser(description="Describe a video with local models.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--model", default="qwen3-vl:8b", help="Ollama vision model (default: qwen3-vl:8b)")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama server URL")
    ap.add_argument("--segment", type=float, default=30, help="seconds per segment (default: 30)")
    ap.add_argument("--frames", type=int, default=4, help="frames sent per segment (default: 4)")
    ap.add_argument("--width", type=int, default=768, help="max frame width in px (default: 768)")
    ap.add_argument("--whisper", default="large-v3-turbo", help="faster-whisper model (default: large-v3-turbo)")
    ap.add_argument("--whisper-device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--language", default=None, help="speech language code, e.g. en, ru, kk (default: detect)")
    ap.add_argument("--no-audio", action="store_true", help="skip speech transcription")
    ap.add_argument("--words", type=int, default=150, help="target length of the final description")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: next to the video)")
    args = ap.parse_args()

    import ollama

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"File not found: {video}")
    out_dir = (args.out or video.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    client = ollama.Client(host=args.host)
    try:
        client.show(args.model)
    except ollama.ResponseError:
        sys.exit(f"Model {args.model} is not pulled. Run: ollama pull {args.model}")
    except Exception as e:
        sys.exit(f"Cannot reach Ollama at {args.host}: {e}\nStart it with: ollama serve")

    duration = probe_duration(video)
    print(f"[video] {video.name}, {fmt_time(duration)}", file=sys.stderr)

    lines, lang = ([], None) if args.no_audio else transcribe(video, args.whisper, args.whisper_device, args.language)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        start = 0.0
        while start < duration:
            end = min(start + args.segment, duration)
            frames = extract_frames(video, start, end - start, args.frames, tmp, args.width)
            speech = transcript_for(lines, start, end)
            block = f'Speech during this part: "{speech}"' if speech else "There is no speech in this part."
            prompt = SEGMENT_PROMPT.format(start=fmt_time(start), end=fmt_time(end), transcript_block=block)

            t0 = time.time()
            text = chat(client, args.model, prompt, images=frames)
            print(f"[{fmt_time(start)}-{fmt_time(end)}] {time.time() - t0:.1f}s", file=sys.stderr)
            results.append({"start": start, "end": end, "speech": speech, "description": text})
            for f in frames:
                f.unlink(missing_ok=True)
            start = end

    notes = "\n".join(f"[{fmt_time(r['start'])}-{fmt_time(r['end'])}] {r['description']}" for r in results)
    language = {"en": "English", "ru": "Russian", "kk": "Kazakh"}.get(lang or args.language or "en", lang or "English")
    summary = chat(client, args.model, SUMMARY_PROMPT.format(language=language, words=args.words, notes=notes), num_ctx=32768)

    stem = video.stem
    json_path = out_dir / f"{stem}.segments.json"
    md_path = out_dir / f"{stem}.description.md"
    json_path.write_text(json.dumps({"video": video.name, "model": args.model, "segments": results},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(f"{summary}\n\n---\n\n## Scene notes\n\n{notes}\n", encoding="utf-8")

    print(summary)
    print(f"\nSaved {md_path}\nSaved {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
