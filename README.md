# Cash desk violation detector — prototype v0

```
frontend/            web page (index.html), served by the backend at /
backend/service/     FastAPI app, pipeline, models wrappers, rules, legal catalog
backend/tools/       evaluation, training, caching and probe scripts
backend/models/      fine-tuned detectors (id/money, key, unlocked laptop)
runtime/             caches, uploads, results (generated)
open/                training videos and labels (from the organisers)
```
All commands are run from the project root.

Local service (no cloud AI): upload a `.mov`, get verdict, violations with timecode, evidence frames
and the RK normative reference. API matches the case spec (`POST /api/v1/analyze`, `GET /api/v1/jobs/{id}`, `X-API-Key`).

## Run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe transformers accelerate fastapi "uvicorn[standard]" python-multipart opencv-python-headless numpy pillow openpyxl sentencepiece protobuf bitsandbytes
API_KEY=<secret> .venv/Scripts/python -X utf8 -m uvicorn service.app:app --app-dir backend --host 0.0.0.0 --port 8000
```

UI: http://localhost:8000 — API: `curl -X POST localhost:8000/api/v1/analyze -H "X-API-Key: <secret>" -F file=@video.mov`.
Model weights download on first run into `HF_HOME` (default `C:/hf_cache`; must be an ASCII path —
`sentencepiece` cannot open files under the Cyrillic user folder). After that no internet is needed.

## How it works

`backend/service/signals.py` (every 0.5 s): RT-DETRv2 (COCO, Apache-2.0) for people/cups/bags/laptop;
OWLv2 (Apache-2.0) once per video for the desk, then on the upscaled desk crop for cash, cash bag,
documents, personal items; SigLIP (Apache-2.0) for laptop screen state (unlocked / lock / off) and safe
state (OWLv2 locates the safe once per video, SigLIP classifies open / closed on an expanded crop).
`backend/service/rules.py`: temporal rules over these signals. `backend/service/rules_catalog.py`: type → regulation.
`backend/service/vlm.py`: Qwen3-VL-4B-Instruct (Apache-2.0, 4-bit, ~4.5 GB VRAM, ~8 GB download) answers narrow
yes/no questions about short clips the rules select. Currently one check: after the cashier walks into an
empty room, did she close the door (catches Х-32). Disable with `VLM=0`.

Why not use the VLM as the main detector: tested open-ended ("check these 10 rules") on 12 videos it flagged
3/5 clean videos with the same hallucinated "person in the doorway" and still missed 4/5 hard cases.
Tested with narrow questions it was reliable only on the door question (ID shown, key in safe,
re-bundled note, cash to pocket: always answered "no", so useless).

## Tuning loop

```bash
.venv/Scripts/python -X utf8 backend/tools/extract_all.py open/open/videos runtime/signals   # ~25 min, cached
.venv/Scripts/python -X utf8 backend/tools/vlm_cache.py open/open/videos runtime/vlm             # targeted VLM checks, ~1 min
.venv/Scripts/python -X utf8 backend/tools/evaluate.py runtime/signals open/open/open_labels.xlsx violation_classification.csv
```

Current (40 unique training videos): verdict accuracy 1.00 (21/21 violations, 0 false positives / 19 clean),
type correct on 20/21, median timecode error 1.5 s (13/21 within 2 s).
**These are training-set numbers.** Every threshold was tuned on these same videos and several rules / detectors
were built from a single example, so expect lower accuracy on the jury's videos (different camera angle).
With `SAFETY_NET=verdict` (every hint counts): 0.80, 18/21 flagged but 5 false positives. Speed ≈ 0.6–1.0× video length on RTX 5050
(142 s video ≈ 93 s with the VLM loaded; ~7 GB of 8 GB VRAM), inside the 5-minute limit.

## Implemented rules

| Type | Rule |
|---|---|
| V1 cash / bag left unattended | nobody seated, room empties, cash or bag still on the desk at the end of the away period |
| V4 documents left | same, documents / cup on the desk |
| V5 workstation unlocked | same, screen classified "open application" by SigLIP, or ≥ 9 % near-white pixels on the screen while away (catches the darkened room in Х-11, where SigLIP says "off"; all other leave events ≤ 5.3 %) |
| V3 previous client's cash | cash seen ≥ 3 times around the client's exit, next person enters (≤ 25 s) and cash is still seen |
| V7 counted without client | client hands over cash ≤ 6 s after sitting down and stands up / leaves ≤ 8 s after the handover |
| V10 bag without check | collector-like visit of 3–12 s during which a cash bag appears |
| V12 third person | ≥ 3 people in the room ≥ 2 s |
| V13 personal items | handbag / jacket on the desk ≥ 3 s |
| V8 suspicious note returned | cashier's box reaches the counter (only where that is unusual for the camera), then within 6 s a moving cash box (static ID cards ignored) goes from the cashier side to the client side. Built from one example (Х-1): overfitting risk |
| V8 notes not checked on detector | lone cashier uses the counter twice (load, take out), and no cash is ever seen on the banknote detector device; only when the device was located as an object different from the counter. Built from Х-44 vs Х-45 |
| V14 drinking | cup raised above the desk line at the face of a person, ≥ 2 consecutive samples |
| V2 safe left open | safe classified open ≥ 9 s, or still open at the end while the cashier sits; only if the safe was located confidently (OWLv2 score ≥ 0.26) |
| V11 door left open | VLM: the first person walking into an empty room does not close the door (asked on the 8 s after entry) |
| V11 outsider | newcomer track ≥ 2.5 s, tall and wide enough (not seen through the glass), static, never sits or comes to the desk, cashier seated when he appears, nobody sits down afterwards |

## Safety net (unseen violation types)

`review_hints()` in `backend/service/rules.py` flags moments the rules cannot judge. The API returns them in an extra
`review_hints` list (same item format as `violations`); the UI shows them under "Требует внимания проверяющего".

| Hint | Fires when | Counts toward verdict |
|---|---|---|
| R1 cashier leaves the room | room empties after the cashier was seated and no other leave-rule fired | no (fires on 5 clean training videos; set `SAFETY_NET=verdict` to count it) |
| R2 cash handled away from the desk | a lone cashier who is not seated holds cash ≥ 10 s; timecode = when the cash disappears | yes (never fired on a clean video; catches Х-13) |

Missed in training set: none. Х-11 is now typed V5 (the team's labels: laptop left unlocked); its safe-ajar part is still not detected.

ID / money detector: RT-DETRv2-R18 (Apache-2.0) fine-tuned with two classes (`id`, `money`) on the team's CVAT
labels for Х-1 (`backend/tools/train_id.py`, weights in `backend/models/idmoney_rtdetr`, ~15 min on RTX 5050), run on the desk
crop of every sampled frame (`backend/service/idcard.py`, cache: `backend/tools/idcard_cache.py`). The `money` class is what makes
it work: trained on `id` alone it took Х-2's blue banknotes for an ID card. Ultralytics YOLO was not used:
AGPL-3.0 conflicts with the commercial-use requirement; the CVAT YOLO 1.1 export is read directly.

| V6 no ID shown | per client session (2 people seated ≥ 6 s, cash seen ≥ 1 s): ID visible < 40 % of the session (Х-2: 0.28; lowest clean: 0.55) |

Key detector: same recipe, one class `key`, trained on the team's CVAT labels for Х-22 (frames ≥ 157; earlier
frames are unlabelled although the key is visible, so they are skipped) + negatives from the end of clean Х-23,
on an x2 crop around the safe (`CROP=safe CLASS_NAME=key MIN_FRAME=157 NEG=X-23:13-18`, weights `backend/models/key_rtdetr`).
The key is normally in the lock while the cashier works (Х-12, Х-13, Х-15, Х-21, Х-24), so the rule is:

| V2 key left in the safe | the room is empty after the cashier left and the median key score over the empty frames ≥ 0.6 (Х-22: 0.96; all other leave events ≤ 0.35) |

Unlocked-laptop detector (`backend/models/unlocked_rtdetr`, from the team's Х-11 labels): **not used** (enable with
`USE_LAPTOP_DET=1`). With one video of positives it learned that camera's laptop, not the screen state: it scores
1.0 on black screens in Х-13 / Х-14 / Х-15 and 0.0 on the unlocked spreadsheet in Х-28. Labels on several videos
with both locked and unlocked screens would be needed. The simple white-pixel test above covers Х-11 instead.

Only one labelled video each: the detectors have not been validated on labelled data from other videos (checked by eye
on Х-3, Х-5, Х-18, Х-19, Х-35, Х-37, Х-39). Labelling ID/money in 3–5 more videos would allow a real test.

Tried and rejected: zero-shot VLM key check (Х-22) — answered "empty" for both Х-22 and Х-23; zero-shot VLM ID check (Х-2) — it says "holding an ID card" for banknotes too.

## Robustness to a different camera angle (simulated)

`backend/tools/augment_eval.py flip warp zoom` re-runs the whole pipeline (all models, rules, VLM) on altered copies of
the 40 unique training videos and stores the results next to the originals, so `backend/tools/evaluate.py
runtime/aug/<variant>/signals …` scores them. ~35 min per variant on the RTX 5050.

| Variant | What it simulates | Before fixes | After fixes |
|---|---|---|---|
| original | — | 1.00 | 1.00 (0 FP) |
| flip | mirrored room (safe / door / counter on the other side) | 0.85 | 0.88 (4 FP, 1 miss) |
| warp | camera turned ~10–15° (perspective) | 0.68 (crashed on zoom, see below) | 0.82 (3 FP, 4 misses) |
| zoom | camera closer and shifted (off-centre 80 % crop) | crash | 0.88 (4 FP, 1 miss) |

Fixed:
- **Crash**: when the desk box reached the image edge the detector-device crop was empty and extraction raised an
  exception (the API would return `failed` on such a video). Crops are now clamped; empty crops return no detections.
- V8 (note returned): the zero-shot cash boxes missed notes in the mirrored view; the fine-tuned `money` detector
  is now used as well, and the last cash-return event is reported (Х-1 timecode now exact in all variants).
- V13 (personal items): the generic COCO "handbag" accepted desk equipment (counting machine) on warped views;
  only bags that arrive after the start and are not on the counter / laptop count now.
- V5 (unlocked screen): SigLIP "unlocked" threshold 0.5 → 0.2 (violations 0.34–0.95 in all variants, clean ≤ 0.08).
- V14 (drinking): cup must not be clearly below the desk line (was: clearly above it).
- V11 (outsider): minimum height 0.17 → 0.16 of the frame; "seated" must hold for most of the track.

Still fragile (need labelled data from more videos / angles, not more threshold tuning):
- V6 no ID: the ID detector learned from Х-1 only; on altered views clean videos fall to 0.29–0.31 ID-visible
  share vs 0.24–0.28 for Х-2 — no margin. Kept the threshold to avoid missing violations.
- V8 detector skipped (Х-44): zero-shot localisation of the banknote detector device fails on every altered view.
- V4 documents (Х-24): zero-shot "documents" score 0.22–0.25 vs 0.29 on a clean video (flip) — no margin.
- V11 / V12 people behind the glass: size thresholds are per-camera; zoom enlarges them into false alarms.
- V14 (Х-46, warp): the raised cup is not detected at all.
