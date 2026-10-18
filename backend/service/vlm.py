"""Targeted checks with a local vision-language model (Qwen3-VL-4B-Instruct, Apache-2.0, 4-bit, ~4.5 GB VRAM).

The VLM is NOT a general violation detector: tested open-ended on the training set it hallucinated
violations on clean videos. It is only asked narrow yes/no questions about short clips that the rules
select, where it was reliable on the training set. Disable with VLM=0.
"""
import os
os.environ.setdefault("HF_HOME", "C:/hf_cache")  # same cache as service/signals.py (ASCII path)
import cv2, torch
from PIL import Image

ENABLED = os.environ.get("VLM", "1") != "0" and torch.cuda.is_available()
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
_m = {}

DOOR_Q = ("CCTV video of a bank cashier's room, frames 1 second apart. After the woman enters the room, does she "
          "close the glass door behind her? Answer yes or no, then one short sentence.")


def _load():
    if not _m:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        _m["proc"] = AutoProcessor.from_pretrained(MODEL)
        _m["model"] = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, quantization_config=q,
                                                                      device_map="cuda").eval()
    return _m


def _frames(path, t0, t1, fps=1.0):
    cap = cv2.VideoCapture(path)
    out, t = [], max(0.0, t0)
    while t <= t1:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        t += 1.0 / fps
    cap.release()
    return out


@torch.no_grad()
def ask(path, t0, t1, question, max_new_tokens=60):
    """Answer a question about the clip [t0, t1] (1 frame / s, full resolution)."""
    m = _load()
    vid = _frames(path, t0, t1)
    if not vid:
        return ""
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
    text = m["proc"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = m["proc"](text=[text], videos=[vid], return_tensors="pt", do_sample_frames=False,
                    video_metadata=[{"fps": 1.0, "total_num_frames": len(vid),
                                     "frames_indices": list(range(len(vid)))}]).to("cuda")
    out = m["model"].generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return " ".join(m["proc"].batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].split())


def door_closed(path, t_entry):
    """Did the cashier close the door after entering at t_entry? -> (closed: bool, raw answer)."""
    ans = ask(path, t_entry + 0.5, t_entry + 8.5, DOOR_Q)
    return not ans.lower().startswith("no"), ans
