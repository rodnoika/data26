# Cashier violation detection: taxonomy + CV plan (Datathon 2026, ForteBank)

Per-video results: [violation_classification.csv](violation_classification.csv). I classified all 22 labeled violations by hand, comparing each one against a "correct" counterpart video from the clean set.

## 1. Violation taxonomy (from the training set)

| ID | Type | Videos | Legal basis (RK) |
|---|---|---|---|
| V1 | Cash/valuables left unattended (cashier leaves room) | Х-15, Х-26 | NBRK Res. No.231, para 20 (cash kept in lockable devices) |
| V2 | Safe left open / not secured | Х-20 (while working), Х-11, Х-22 (on leaving) | No.231 para 20, 80 |
| V3 | Previous client's cash still on desk when next client enters | Х-4, Х-47, Х-38 | No.231 para 20, 22 |
| V4 | Documents/items left on desk when leaving | Х-24 | No.231 para 12, 20; Law On banks art.50 |
| V5 | Workstation left unlocked | Х-28 | ARDFM Res. No.53 (03.04.2026), which replaced NBRK Res. No.48 para 71(9). **Check current paragraph** |
| V6 | Cash accepted with no client ID check | Х-2 | No.231 para 7, 19(1); AML Law No.191-IV art.5 |
| V7 | Cash counted without the client present | Х-34 | No.231 para 20, 35 |
| V8 | Suspicious banknote (from reject pocket) returned or re-bundled | Х-1, Х-44 | No.231 para 59, 61 |
| V9 | Cash moved toward clothing (possible misappropriation) | Х-13 | No.231 para 12 |
| V10 | Collector's bag accepted without document/seal check | Х-17 | No.231 para 24, 106 |
| V11 | Cash-room door left open / outsider enters | Х-30, Х-32, Х-33 | NBRK Res. No.14 (24.02.2020) para 16(1), 19(1); No.231 para 75 |
| V12 | Third person in the service zone during an operation | Х-18 | Law On banks No.2444 art.50 |
| V13 | Personal belongings (bag, coat) on the cash desk | Х-40 | No.231 para 12 |
| V14 | Food/drink at the cash workplace | Х-46 | No.231 para 12 |

Sources:
- **NBRK Res. No.231** of 29.11.2019, *Правила осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей* (reg. No.19680).
- **NBRK Res. No.14** of 24.02.2020, *Правила организации охраны и устройства помещений…* (reg. No.20075).
- **Law RK of 28.08.2009 No.191-IV** on AML/CFT.
- **Law RK of 31.08.1995 No.2444** *О банках и банковской деятельности*.

Paragraph numbers for No.231 and No.14 come from the consolidated text on zakon.uchet.kz. Re-check them on adilet.zan.kz before submitting.

**Confidence notes:**
- V8 (Х-44) is low confidence. V9 and V12 are medium.
- For V9, the service must only say *what is visible* ("cash moved toward clothing"). It must never say "theft". This follows ТЗ §7.
- Dataset quirk: Х-4 and Х-47 are the same file (identical MD5) but are labeled at 57 s and 68 s. Х-5 and Х-48 are also the same file. Deduplicate before any train/val split.

## 2. What the CV system must perceive

**Objects** (detector, fine-tuned on frames from this dataset and auto-labeled with Grounding DINO / OWLv2):
- person
- banknote / cash bundle
- ID card / passport
- cash-in-transit bag (green)
- safe, plus a **safe-door state** (open / closed): a small classifier on the safe crop
- cash-room door, plus a **door state** (open / closed)
- banknote counter, plus its **reject-pocket** region
- laptop, plus a **screen state** (unlocked app / lock screen / off): classify by brightness and content of the screen crop
- cup / bottle, handbag, outerwear, phone, document folder

**Roles** (faces are forbidden, so no face recognition): assign roles from geometry and tracks.
- cashier: the track that sits in the chair behind the counter/laptop
- client: the person on the opposite side of the desk
- collector: a person carrying the green bag
- anyone else in the room is a third person

**Zones**, detected automatically for each video so the system copes with the new camera angle in the final test:
- desk surface
- cashier seat
- client seat
- safe
- door / room exit

**Hands / pose** (RTMPose, Apache-2.0): wrist positions relative to the safe, the counter, the waist (for V9), and handovers between people.

**Tracking**: ByteTrack, giving person tracks and cash-on-desk persistence.

## 3. Rules engine (temporal state machine over 5 fps signals)

| Rule | Trigger, and the timecode to report |
|---|---|
| V1/V4 | Cashier track leaves the room (or ROI) **and** cash/bag/documents are still on the desk → t = exit moment |
| V2 | Safe door open while the cashier is ≥ N s away from the safe, or seated counting → t = first such frame |
| V3 | Client A leaves, cash is still on the desk, and a new person enters → t = entry of the new person |
| V5 | Cashier leaves and the screen state is still "unlocked" → t = exit |
| V6 | Cash handover client→cashier with no ID-card object handed over earlier in the session → t = cash handover |
| V7 | Cash on the counter/desk while the client track has left → t = client exit |
| V8 | Notes taken from the reject pocket go back to the client or into the bundle → t = handover |
| V9 | Cashier's wrist holding cash ends near the waist/pocket and the cash disappears (not to safe or desk) → t = that frame |
| V10 | Bag handed over by the collector, with no document exchange and no bag-inspection pose within X s → t = bag on desk |
| V11 | Door state stays "open" for more than N s after the cashier enters, or a non-role person enters → t = event |
| V12 | Third person within the desk zone while a client session is active → t = approach |
| V13/V14 | Handbag/outerwear/cup detected on the desk zone for more than N s → t = first stable detection |

Each fired rule outputs:
- 1–3 evidence frames: the trigger frame ± 1 s
- a factual description template
- a rule reference from a lookup table (rule ID → document, paragraph)

Omission violations (V5, V6, V10, V11) are timestamped at the moment the obligation was breached.

Bias toward recall (ТЗ FAQ 16 says a missed violation is unacceptable): use low thresholds per rule, then require the signal to persist for k frames to cut false alarms.

## 4. Implementation steps

1. **Data**:
   - Deduplicate the training set.
   - Extract frames at 5 fps.
   - Auto-label with Grounding DINO / OWLv2 and correct by hand (a few hundred frames).
   - Add your own staged clips filmed from other angles, plus horizontal flips.
2. **Models**:
   - Detector: RT-DETR or RTMDet (Apache-2.0). Avoid Ultralytics YOLO because of AGPL-3.0: the case requires licenses that allow commercial use.
   - State classifiers: SigLIP/CLIP zero-shot first, fine-tuned later.
   - Pose: RTMPose.
3. **Rules engine**: as in §3, with thresholds tuned on the labeled timecodes (score: video-level verdict plus timecode accuracy).
4. **Service**:
   - FastAPI with `POST /api/v1/analyze` and `GET /api/v1/jobs/{id}` behind an `X-API-Key` header, plus a background worker.
   - Simple web UI.
   - Fully offline: weights baked into the Docker image.
   - Budget: under 5 min per video. A 145 s clip at 5 fps is about 725 frames, which runs on CPU and is fast on GPU.
5. **Robustness**: zones come from detections, not fixed polygons. Check on the second camera angle (Х-30/32/33 and similar clips) as a stand-in for the unseen test angle.
