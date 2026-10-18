"""VLM on an upscaled still crop: usage probe_vlm_crop.py "<question>" name:t:x0:y0:x1:y1 ... (crop in original px)"""
import sys, os, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, torch
from PIL import Image
from service import vlm


@torch.no_grad()
def ask_image(img, q):
    m = vlm._load()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    text = m["proc"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = m["proc"](text=[text], images=[img], return_tensors="pt").to("cuda")
    out = m["model"].generate(**inp, max_new_tokens=40, do_sample=False)
    return " ".join(m["proc"].batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].split())


if __name__ == "__main__":
    q = sys.argv[1]
    for spec in sys.argv[2:]:
        name, t, x0, y0, x1, y1 = spec.split(":")
        path = [p for p in glob.glob("open/open/videos/*.mov") if os.path.basename(p).replace("Х", "X")[:-4] == name][0]
        cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000); ok, fr = cap.read()
        c = fr[int(y0):int(y1), int(x0):int(x1)]
        c = cv2.resize(c, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        st = time.time()
        print(f"{name}@{t}: {ask_image(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)), q)}  ({time.time() - st:.1f}s)", flush=True)
