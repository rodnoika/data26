"""SigLIP zero-shot on crops. usage: probe_siglip.py out_dir video:sec:x0:y0:x1:y1 ...  (crop in original px)"""
import sys, os
import cv2, torch, numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

PROMPTS = {
    "desk": ["a photo of an empty wooden desk with a laptop and a phone",
             "a photo of banknotes lying on a wooden desk",
             "a photo of stacks of cash money on a desk",
             "a photo of a green bag on a desk",
             "a photo of a handbag and a jacket on a desk",
             "a photo of documents and a cup on a desk"],
}
name = "google/siglip-so400m-patch14-384"
proc = AutoProcessor.from_pretrained(name)
model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to("cuda").eval()
txt = PROMPTS["desk"]
ti = proc(text=txt, padding="max_length", return_tensors="pt").to("cuda")
with torch.no_grad():
    tf = model.get_text_features(**ti); tf = getattr(tf, "pooler_output", tf); tf = tf / tf.norm(dim=-1, keepdim=True)
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for spec in sys.argv[2:]:
    path, sec, x0, y0, x1, y1 = spec.rsplit(":", 5)
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000)
    ok, fr = cap.read()
    crop = fr[int(y0):int(y1), int(x0):int(x1)]
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ii = proc(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        f = model.get_image_features(pixel_values=ii["pixel_values"].half()); f = getattr(f, "pooler_output", f); f = f / f.norm(dim=-1, keepdim=True)
    logits = (f @ tf.T)[0].float() * model.logit_scale.exp().float() + model.logit_bias.float()
    p = torch.sigmoid(logits).detach().cpu().numpy()
    tag = os.path.basename(path).replace("Х", "X").replace(".mov", "") + f"_{sec}"
    open(os.path.join(out, tag + ".jpg"), "wb").write(cv2.imencode(".jpg", crop)[1].tobytes())
    print(tag, " ".join(f"{i}:{v:.3f}" for i, v in enumerate(p)), flush=True)
