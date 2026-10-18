"""Feasibility test: can a local VLM (Qwen3-VL-4B, 4-bit) spot cash-desk violations the rules miss?

usage: probe_vlm.py name[:t0-t1] ...   (window in seconds; default = whole video)
Frames are sampled at FPS from the window and sent as one video. Prints the model's JSON answer and timing.
"""
import sys, os, glob, json, time, re
os.environ.setdefault("HF_HOME", "C:/hf_cache")
import cv2, torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
FPS = float(os.environ.get("VLM_FPS", "1.0"))
SIZE = (252, 448)  # w, h — keeps the 464x848 portrait aspect

PROMPT = """You are checking a CCTV video of a bank cashier's cash room (top-down camera, one desk, a grey safe, a banknote counting machine, a laptop). Frames are 1 second apart; frame k is at time {t0}+k seconds.

Check these rules and report ONLY what is clearly visible:
1. Client must show an ID document before handing over cash.
2. Cash must be counted while the client is present.
3. Suspicious banknotes rejected by the counting machine must not be returned to the client or put back into the bundle.
4. The safe must be closed and locked, key removed, when the cashier is not using it or leaves.
5. The cash room door must be closed; no outsiders inside or standing in the doorway.
6. Cash, cash bags and documents must not be left unattended on the desk.
7. The laptop must be locked when the cashier leaves.
8. No personal bags, coats, food or drinks at the cash desk.
9. Cash must not be put into the cashier's pockets or clothing.
10. A collector's cash bag must be accepted only after checking documents and the seal.

Answer with JSON only:
{{"violation": true|false, "rule": <number or null>, "time_sec": <number or null>, "what_is_visible": "<one sentence>"}}"""

_m = {}


def load():
    if not _m:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        _m["proc"] = AutoProcessor.from_pretrained(MODEL)
        _m["model"] = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, quantization_config=q, device_map="cuda").eval()
    return _m


def frames(path, t0, t1):
    cap = cv2.VideoCapture(path)
    out, t = [], t0
    while t <= t1:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        out.append(Image.fromarray(cv2.cvtColor(cv2.resize(fr, SIZE), cv2.COLOR_BGR2RGB)))
        t += 1.0 / FPS
    return out


@torch.no_grad()
def ask(path, t0, t1):
    m = load()
    vid = frames(path, t0, t1)
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPT.format(t0=t0)}]}]
    text = m["proc"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = m["proc"](text=[text], videos=[vid], return_tensors="pt", do_sample_frames=False,
                    video_metadata=[{"fps": FPS, "total_num_frames": len(vid),
                                     "frames_indices": list(range(len(vid)))}]).to("cuda")
    out = m["model"].generate(**inp, max_new_tokens=160, do_sample=False)
    ans = m["proc"].batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return ans, len(vid), inp["input_ids"].shape[1]


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        name, _, win = spec.partition(":")
        path = [p for p in glob.glob("open/open/videos/*.mov") if os.path.basename(p).replace("Х", "X")[:-4] == name][0]
        cap = cv2.VideoCapture(path); dur = cap.get(7) / cap.get(5)
        t0, t1 = (float(x) for x in win.split("-")) if win else (0.0, dur - 0.2)
        t = time.time()
        ans, nf, ntok = ask(path, t0, min(t1, dur - 0.2))
        print(f"{name} [{t0:.0f}-{t1:.0f}s] frames={nf} tokens={ntok} {time.time() - t:.1f}s -> {" ".join(ans.split())}", flush=True)
