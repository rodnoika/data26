"""Time each stage of signal extraction on one frame (repeated)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, torch
from service import signals as S

cap = cv2.VideoCapture(sys.argv[1]); cap.set(cv2.CAP_PROP_POS_MSEC, 5000); ok, fr = cap.read()
m = S._load()
for _ in range(2):
    S._rtdetr(fr); S._owl(fr[250:560, 0:350], S.DESK_Q, 0.15)
torch.cuda.synchronize()
for name, fn in [("rtdetr", lambda: S._rtdetr(fr)), ("owl_desk", lambda: S._owl(fr[250:560, 0:350], S.DESK_Q, 0.15))]:
    t = time.time()
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    print(name, f"{(time.time() - t) * 100:.0f} ms/frame")
from PIL import Image
img = Image.fromarray(fr[250:560, 0:350])
t = time.time()
for _ in range(10):
    m["op"](text=[S.DESK_Q], images=img, return_tensors="pt")
print("owl_preproc", f"{(time.time() - t) * 100:.0f} ms/frame")
t = time.time()
for _ in range(10):
    m["rp"](images=Image.fromarray(fr), return_tensors="pt")
print("rt_preproc", f"{(time.time() - t) * 100:.0f} ms/frame")
