"""Targeted VLM questions on short full-resolution clips: is the VLM useful as a second check on one moment?
Each case: video, window, question, expected answer for the violation video vs. its clean twin."""
import sys, os, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_vlm as V

V.SIZE = (464, 848)   # full resolution
V.FPS = 1.0

CASES = [
    # (video, t0, t1, question)
    ("X-2", 5, 13, "Before handing over the cash, does the client show or hand an ID card / passport to the cashier? Answer yes or no, then one short sentence."),
    ("X-37", 3, 12, "Before handing over the cash, does the client show or hand an ID card / passport to the cashier? Answer yes or no, then one short sentence."),
    ("X-3", 5, 14, "Before handing over the cash, does the client show or hand an ID card / passport to the cashier? Answer yes or no, then one short sentence."),
    ("X-22", 9, 18, "Look at the grey metal safe at the end of the clip. Is a key left in the safe's lock, or is the safe door open? Answer yes or no, then one short sentence."),
    ("X-23", 8, 17, "Look at the grey metal safe at the end of the clip. Is a key left in the safe's lock, or is the safe door open? Answer yes or no, then one short sentence."),
    ("X-32", 4, 12, "After the woman enters the room, does she close the glass door behind her? Answer yes or no, then one short sentence."),
    ("X-31", 4, 12, "After the woman enters the room, does she close the glass door behind her? Answer yes or no, then one short sentence."),
    ("X-44", 11, 19, "Does the cashier take banknotes out of the counting machine's reject pocket and put them back into the stack by hand? Answer yes or no, then one short sentence."),
    ("X-45", 10, 18, "Does the cashier take banknotes out of the counting machine's reject pocket and put them back into the stack by hand? Answer yes or no, then one short sentence."),
    ("X-13", 43, 50, "Does the woman put banknotes into her pocket or clothing? Answer yes or no, then one short sentence."),
    ("X-14", 40, 48, "Does the woman put banknotes into her pocket or clothing? Answer yes or no, then one short sentence."),
]

for name, t0, t1, q in CASES:
    V.PROMPT = ("CCTV video of a bank cashier's room, frames 1 second apart starting at {t0} s. " + q).replace("{", "{{").replace("}", "}}").replace("{{t0}}", "{t0}")
    path = [p for p in glob.glob("open/open/videos/*.mov") if os.path.basename(p).replace("Х", "X")[:-4] == name][0]
    t = time.time()
    ans, nf, ntok = V.ask(path, t0, t1)
    print(f"{name:5s} [{t0}-{t1}s] tok={ntok} {time.time() - t:.1f}s -> {' '.join(ans.split())[:160]}", flush=True)
