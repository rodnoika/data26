"""FastAPI service: upload a .mov, get violations with timecodes, evidence frames, rule refs.

Run (from the project root):
      .venv/Scripts/python -m uvicorn service.app:app --app-dir backend --host 0.0.0.0 --port 8000
Env:  API_KEY (required for /api/v1/*), DATA_DIR (default <project>/runtime)
"""
import os, uuid, queue, threading, traceback, json, time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse

API_KEY = os.environ.get("API_KEY", "dev-key-change-me")
PROJECT = Path(__file__).resolve().parents[2]          # <project>/backend/service/app.py
FRONTEND = PROJECT / "frontend"
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT / "runtime"))
UPLOADS = DATA_DIR / "uploads"; RESULTS = DATA_DIR / "results"
UPLOADS.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
MAX_BYTES = 200 * 1024 * 1024

app = FastAPI(title="Cash desk violation detector")
jobs: dict[str, dict] = {}
work_q: "queue.Queue[str]" = queue.Queue()


def _check_key(key: str | None):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _worker():
    from service.pipeline import analyze  # heavy import: load models once, in the worker
    while True:
        job_id = work_q.get()
        job = jobs[job_id]
        try:
            t0 = time.time()
            result = analyze(job["path"], progress=lambda p: job.update(progress=round(p, 2)))
            job.update(status="done", verdict=result["verdict"], violations=result["violations"],
                       review_hints=result.get("review_hints", []), elapsed_sec=round(time.time() - t0, 1))
        except Exception as e:  # never leave a job hanging in "processing"
            traceback.print_exc()
            job.update(status="failed", error=str(e))
        (RESULTS / f"{job_id}.json").write_text(json.dumps(_public(job), ensure_ascii=False), encoding="utf-8")
        try:
            os.remove(job["path"])
        except OSError:
            pass


def _public(job: dict) -> dict:
    out = {"job_id": job["job_id"], "status": job["status"]}
    if job["status"] == "done":
        out.update(verdict=job["verdict"], violations=job["violations"], review_hints=job.get("review_hints", []))
    elif job["status"] == "failed":
        out["error"] = job.get("error", "")
    else:
        out["progress"] = job.get("progress", 0.0)
    return out


@app.on_event("startup")
def _start():
    threading.Thread(target=_worker, daemon=True).start()


async def _save_upload(file: UploadFile) -> dict:
    job_id = uuid.uuid4().hex[:8]
    path = UPLOADS / f"{job_id}.mov"
    size = 0
    with open(path, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                f.close(); path.unlink(missing_ok=True)
                raise HTTPException(413, "file too large")
            f.write(chunk)
    job = dict(job_id=job_id, status="processing", path=str(path), filename=file.filename, progress=0.0)
    jobs[job_id] = job
    work_q.put(job_id)
    return {"job_id": job_id, "status": "processing"}


@app.post("/api/v1/analyze")
async def analyze_api(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    _check_key(x_api_key)
    return await _save_upload(file)


@app.get("/api/v1/jobs/{job_id}")
def job_api(job_id: str, x_api_key: str | None = Header(None)):
    _check_key(x_api_key)
    return _job_or_404(job_id)


def _job_or_404(job_id: str) -> dict:
    if job_id in jobs:
        return _public(jobs[job_id])
    p = RESULTS / f"{job_id}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    raise HTTPException(404, "job not found")


# --- browser UI (no key: it is the human-facing page) ---
@app.post("/ui/analyze")
async def analyze_ui(file: UploadFile = File(...)):
    return await _save_upload(file)


@app.get("/ui/jobs/{job_id}")
def job_ui(job_id: str):
    return _job_or_404(job_id)


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/health")
def health():
    return JSONResponse({"ok": True, "queue": work_q.qsize()})
