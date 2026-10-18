"""Fine-tune RT-DETRv2-R18 (Apache-2.0) as a one-class ID-card detector from CVAT "YOLO 1.1" labels.

usage: train_id.py <labels_dir with obj_Train_data/> <video.mov> <out_dir>
Training images are crops of the desk area (desk box from the cached signals, extended upwards to include
hands), resized to 640x640, so the ~18 px card becomes ~40 px. Same crop is used at inference (service/idcard.py).
"""
import sys, os, glob, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "C:/hf_cache")
import cv2, numpy as np, torch
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from service.idcard import desk_crop_box, safe_crop_box, laptop_crop_box, SIZE

lab_dir, video, out = sys.argv[1:4]
name = os.path.basename(video).replace("Х", "X")[:-4]
sig = json.load(open(f"runtime/signals/{name}.json", encoding="utf-8"))
H, W = sig["height"], sig["width"]
CROP = os.environ.get("CROP", "desk")            # "desk" (id / money) or "safe" (key in the lock)
CLASS_NAME = os.environ.get("CLASS_NAME", "id")
MIN_FRAME = int(os.environ.get("MIN_FRAME", 0))  # frames before it are unlabelled, not "empty"
crop_of = lambda sg: {"safe": lambda: safe_crop_box(sg["layout"]["safe"]["box"], H, W),
                      "laptop": lambda: laptop_crop_box(sg["layout"]["laptop"]["box"], H, W)}.get(
    CROP, lambda: desk_crop_box(sg["layout"]["desk"]["box"], H, W))()
crop = crop_of(sig)

MONEY_DIR = os.environ.get("MONEY_DIR", "")
CLASSES = ["id", "money"] if MONEY_DIR else [os.environ.get("CLASS_NAME", "id")]
# ---- load frames + labels (every 2nd frame: neighbours are near-duplicates)
cap = cv2.VideoCapture(video)
data = []
for f in sorted(glob.glob(os.path.join(lab_dir, "obj_Train_data", "*.txt"))):
    n = int(os.path.basename(f)[6:12])
    if n % 2 or n < MIN_FRAME:
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, n); ok, fr = cap.read()
    if not ok:
        continue
    boxes = []
    # class 0 = id (lab_dir); class 1 = money (MONEY_DIR, same video, CVAT export of the same frames)
    srcs = [(f, 0)] + ([(os.path.join(MONEY_DIR, "obj_Train_data", os.path.basename(f)), 1)] if MONEY_DIR else [])
    for path_, cls in srcs:
        if not os.path.exists(path_):
            continue
        for line in open(path_).read().strip().splitlines():
            _, cx, cy, w, h = map(float, line.split())
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H, cls])
    data.append((n, fr, boxes))
# ---- hard negatives: NEG=X-10,X-14 (whole video) or X-23:13-18 (time window, s)
crops = {name: crop}
for spec in [n for n in os.environ.get("NEG", "").split(",") if n]:
    neg, _, win = spec.partition(":")
    vp = [p for p in glob.glob(os.path.join(os.path.dirname(video), "*.mov")) if os.path.basename(p).replace("Х", "X")[:-4] == neg][0]
    ns = json.load(open(f"runtime/signals/{neg}.json", encoding="utf-8"))
    crops[neg] = crop_of(ns)
    t, t_end = (float(x) for x in win.split("-")) if win else (0.0, ns["duration"])
    c2 = cv2.VideoCapture(vp)
    while t < min(t_end, ns["duration"]):
        c2.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, fr = c2.read()
        if not ok:
            break
        data.append((neg, fr, [])); t += 0.2 if win else 0.5
print(f"{len(data)} frames, {sum(bool(b) for _, _, b in data)} with a card", flush=True)


def sample(fr, boxes, aug, key=name):
    x0, y0, x1, y1 = crops.get(key, crop)
    if aug:   # jitter the crop so the card is not always at the same position / scale
        w, h = x1 - x0, y1 - y0
        x0 += random.uniform(-.15, .15) * w; x1 += random.uniform(-.15, .15) * w
        y0 += random.uniform(-.15, .15) * h; y1 += random.uniform(-.15, .15) * h
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    img = cv2.resize(fr[y0:y1, x0:x1], (SIZE, SIZE))
    sx, sy = SIZE / (x1 - x0), SIZE / (y1 - y0)
    bb = []
    for b in boxes:
        c = [max(b[0], x0), max(b[1], y0), min(b[2], x1), min(b[3], y1)]
        if c[2] - c[0] > 3 and c[3] - c[1] > 3:
            bb.append([(c[0] - x0) * sx, (c[1] - y0) * sy, (c[2] - x0) * sx, (c[3] - y0) * sy, b[4] if len(b) > 4 else 0])
    if aug:
        if random.random() < .5:
            img = img[:, ::-1].copy(); bb = [[SIZE - b[2], b[1], SIZE - b[0], b[3], b[4]] for b in bb]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(.6, 1.4); hsv[..., 2] *= random.uniform(.7, 1.3)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        if random.random() < .3:
            img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, bb


def to_batch(items):
    px = torch.stack([torch.from_numpy(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255 for i, _ in items])
    labels = []
    for _, bb in items:
        cls = torch.tensor([x[4] for x in bb], dtype=torch.long)
        b = torch.tensor([x[:4] for x in bb], dtype=torch.float32).reshape(-1, 4) / SIZE
        cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], -1)
        labels.append({"class_labels": cls.cuda(), "boxes": cxcywh.cuda()})
    return px.cuda(), labels


model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd", num_labels=len(CLASSES),
                                                   id2label=dict(enumerate(CLASSES)), label2id={c: i for i, c in enumerate(CLASSES)}, ignore_mismatched_sizes=True).cuda().train()
opt = torch.optim.AdamW([{"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": 1e-5},
                         {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": 1e-4}],
                        weight_decay=1e-4)
EPOCHS, BS = int(os.environ.get("EPOCHS", 25)), 8
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * (len(data) // BS + 1))
for ep in range(EPOCHS):
    random.shuffle(data); tot = 0
    for i in range(0, len(data), BS):
        px, labels = to_batch([sample(fr, b, True, k if isinstance(k, str) else name) for k, fr, b in data[i:i + BS]])
        loss = model(pixel_values=px, labels=labels).loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1); opt.step(); sched.step()
        tot += loss.item()
    print(f"epoch {ep + 1}/{EPOCHS} loss {tot / (len(data) / BS):.3f}", flush=True)

os.makedirs(out, exist_ok=True)
model.save_pretrained(out)
RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd").save_pretrained(out)
print("saved", out)
