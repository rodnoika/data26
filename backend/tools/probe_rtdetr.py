"""Check COCO RT-DETRv2 detections on a few frames. usage: probe_rtdetr.py out_dir video:sec ..."""
import sys, os, time
import cv2, torch
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
name = "PekingU/rtdetr_v2_r50vd"
proc = RTDetrImageProcessor.from_pretrained(name)
model = RTDetrV2ForObjectDetection.from_pretrained(name).to("cuda").eval().half()
id2l = model.config.id2label

for spec in sys.argv[2:]:
    path, sec = spec.rsplit(":", 1)
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000)
    ok, fr = cap.read()
    img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    t = time.time()
    inp = proc(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = model(pixel_values=inp["pixel_values"].half())
    res = proc.post_process_object_detection(o, threshold=0.3, target_sizes=[img.size[::-1]])[0]
    dt = time.time() - t
    vis = fr.copy()
    for b, s, l in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist()):
        x0, y0, x1, y1 = map(int, b)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.putText(vis, f"{id2l[l]} {s:.2f}", (x0, max(10, y0 - 2)), 0, 0.4, (0, 0, 255), 1)
    tag = os.path.basename(path).replace("Х", "X").replace(".mov", "") + f"_{sec}"
    open(os.path.join(out, tag + ".jpg"), "wb").write(cv2.imencode(".jpg", vis)[1].tobytes())
    print(tag, f"{dt*1000:.0f}ms", [f"{id2l[l]}:{s:.2f}" for s, l in zip(res["scores"].tolist(), res["labels"].tolist())], flush=True)
