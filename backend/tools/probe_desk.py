"""RT-DETR finds the desk (diningtable overlapping the laptop), OWLv2 runs on the upscaled desk crop.
usage: probe_desk.py out_dir video:sec ..."""
import sys, os
import cv2, torch
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection, Owlv2Processor, Owlv2ForObjectDetection

Q = ["banknotes", "paper money", "a stack of banknotes", "a green bag", "a cup", "a black folder",
     "a handbag", "a jacket", "an id card", "a phone"]
dev = "cuda"
rp = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd")
rm = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd").to(dev).eval().half()
op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
om = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", torch_dtype=torch.float16).to(dev).eval()
L = rm.config.id2label


def iou_area(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for spec in sys.argv[2:]:
    path, sec = spec.rsplit(":", 1)
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000)
    ok, fr = cap.read(); H, W = fr.shape[:2]
    img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        o = rm(pixel_values=rp(images=img, return_tensors="pt")["pixel_values"].to(dev).half())
    r = rp.post_process_object_detection(o, threshold=0.3, target_sizes=[(H, W)])[0]
    dets = [(L[l], s, b) for l, s, b in zip(r["labels"].tolist(), r["scores"].tolist(), r["boxes"].tolist())]
    laps = [b for n, s, b in dets if n == "laptop"]
    tables = [b for n, s, b in dets if n == "diningtable"]
    # desk = table overlapping laptop most; fallback: the highest table
    desk = max(tables, key=lambda t: max([iou_area(t, l) for l in laps] + [0]) * 1e6 - t[1]) if tables else [0, H * .3, W, H * .7]
    x0, y0, x1, y1 = map(int, desk)
    y0 = max(0, y0 - int(0.08 * H))  # include objects standing on the desk
    crop = fr[y0:y1, x0:x1]
    ci = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    inp = op(text=[Q], images=ci, return_tensors="pt").to(dev); inp["pixel_values"] = inp["pixel_values"].half()
    with torch.no_grad():
        oo = om(**inp)
    S = max(ci.size)
    prob = torch.sigmoid(oo.logits[0].float()); sc, lab = prob.max(-1); keep = sc > 0.15
    cx, cy, w, h = (oo.pred_boxes[0][keep].float() * S).unbind(-1)
    vis = crop.copy()
    found = []
    for (bx0, by0, bx1, by1), s, l in zip(torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], -1).tolist(), sc[keep].tolist(), lab[keep].tolist()):
        cv2.rectangle(vis, (int(bx0), int(by0)), (int(bx1), int(by1)), (0, 255, 0), 1)
        cv2.putText(vis, f"{Q[l][:12]} {s:.2f}", (int(bx0), max(10, int(by0) - 2)), 0, 0.4, (0, 0, 255), 1)
        found.append(f"{Q[l]}:{s:.2f}")
    tag = os.path.basename(path).replace("Х", "X").replace(".mov", "") + f"_{sec}"
    open(os.path.join(out, tag + ".jpg"), "wb").write(cv2.imencode(".jpg", vis)[1].tobytes())
    print(tag, (x0, y0, x1, y1), sorted(found, key=lambda z: -float(z.split(':')[1]))[:8], flush=True)
