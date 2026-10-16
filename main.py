"""
CHiLLER — FastAPI server (controller layer).

Wires the model manager (models.py) and inference primitives (inference.py) to HTTP routes.
Config lives in config.py, prompts in prompts.py, schemas in schemas.py, GPU selection in gpu.py.

Models load to GPU on first request and offload after IDLE_TIMEOUT seconds; 503 if GPU is unavailable.
"""

import asyncio
from datetime import datetime, timezone
from typing import Annotated, Any, NoReturn

import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import FEEDBACK_MODEL, IDLE_TIMEOUT, LOG_DIR
from feedback_proc import fb_mgr
from logging_setup import _log, log
from models import mgr
from prompts import GRADE_SYSTEM, GRADE_USER
from inference import _candidate, _compute_attribution, _generate_feedback, _score
from schemas import (
    AttributionRequest, AttributionResponse, CorrectionRequest,
    FeedbackRequest, FeedbackResponse, GradeRequest, GradeResponse,
)

app = FastAPI(title="CHiLLER")

# One GPU inference at a time — others get an immediate 503
_gpu_sem = asyncio.Semaphore(1)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Dependencies / guards ───────────────────────────────────────────────────────

def _gpu_503() -> NoReturn:
    raise HTTPException(503, "GPU resources unavailable. Please try again in a moment.")


def _busy_503() -> NoReturn:
    raise HTTPException(503, "Model is busy — please try again in a moment.")


async def require_model() -> None:
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, mgr.ensure_loaded)
    except RuntimeError as e:
        log.error(f"Model load failed: {e}")
        _gpu_503()


async def _run_locked(fn, *args) -> Any:
    """Busy-check + one-at-a-time GPU semaphore + thread-pool offload.

    Deliberately does NOT call require_model() — that stays a per-route
    Depends(require_model) so test_main.py's dependency_overrides keep working.
    """
    if _gpu_sem.locked():
        _busy_503()
    async with _gpu_sem:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn, *args)


# ── Sync inference workers (run in thread pool) ───────────────────────────────

def _grade_core(question: str, response: str, M: int, temperature: float):
    """Score grade candidates 0..M → (pred, confidence, prompt).

    This is the method: we never *generate* a grade — we score the log-likelihood of each
    fixed JSON candidate {"grade": g, "max_grade": M}, temperature-scale with T, softmax to a
    distribution over grades, and take the argmax grade + its probability as the confidence.
    """
    msgs = [
        {"role": "system", "content": GRADE_SYSTEM},
        {"role": "user",   "content": GRADE_USER.format(q=question, a=response, max_g=M)},
    ]
    prompt = mgr.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    z  = np.array([_score(prompt, _candidate(g, M)) for g in range(M + 1)], dtype=np.float64)
    zT = z / temperature
    p  = np.exp(zT - zT.max()); p /= p.sum()
    return int(p.argmax()), float(p.max()), prompt


def _run_grade(req: GradeRequest) -> GradeResponse:
    try:
        M = req.max_grade
        pred, conf, prompt = _grade_core(req.question, req.response, M, req.temperature)
        mgr.touch()

        _log({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "grade",
            "question": req.question, "response": req.response,
            "max_grade": M, "predicted_grade": pred,
            "confidence": round(conf, 4), "accepted": conf >= req.tau,
            "tau": req.tau, "temperature": req.temperature,
        })

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        log.error(f"OOM during grading: {e}")
        mgr.offload()
        _gpu_503()

    return GradeResponse(
        predicted_grade=pred,
        max_grade=M,
        confidence=round(conf, 4),
        accepted=conf >= req.tau,
        prompt=prompt,
    )


def _run_attribution(req: AttributionRequest) -> AttributionResponse:
    try:
        tokens = _compute_attribution(req.prompt, req.response, req.grade, req.max_grade)
        mgr.touch()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        log.error(f"OOM during attribution: {e}")
        mgr.offload()
        _gpu_503()
    return AttributionResponse(tokens=tokens)


def _run_feedback(req: FeedbackRequest) -> FeedbackResponse:
    try:
        mgr.ensure_retriever()
        r = _generate_feedback(req.question, req.response, req.grade, req.max_grade)
        mgr.touch()
        _log({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "feedback",
            "question": req.question, "response": req.response,
            "grade": req.grade, "max_grade": req.max_grade,
            "feedback": r["feedback"], "grounded": r["grounded"],
            "top_score": r["top_score"],
            "cited": [e["label"] for e in r["excerpts"] if e["cited"]],
        })
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        log.error(f"OOM during feedback: {e}")
        mgr.offload()
        fb_mgr.offload()
        _gpu_503()
    return FeedbackResponse(**r)


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    def _free_mb(idx):
        try:
            import pynvml
            pynvml.nvmlInit()
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(idx))
            out = (int(info.free) // 1024 ** 2, int(info.total) // 1024 ** 2)
            pynvml.nvmlShutdown()
            return out
        except Exception:
            return (0, 0)

    mem_free, mem_total = _free_mb(mgr.gpu) if mgr.loaded else (0, 0)
    return {
        "model":        "ready" if mgr.loaded else "shadow",
        "gpu_busy":     _gpu_sem.locked(),
        "grader_gpu":   mgr.gpu if mgr.loaded else None,
        "gpu_mem_free_mb":  mem_free,
        "gpu_mem_total_mb": mem_total,
        "idle_timeout": IDLE_TIMEOUT,
        "log_file":     str(LOG_DIR / "app.log"),
        "feedback_model":  FEEDBACK_MODEL or "grader-base",
        "feedback_loaded": fb_mgr.loaded,
        "feedback_gpu":    fb_mgr.gpu if fb_mgr.loaded else None,
    }


@app.post("/grade", response_model=GradeResponse)
async def grade(req: GradeRequest, _: Annotated[None, Depends(require_model)]):
    if not req.question.strip() or not req.response.strip():
        raise HTTPException(400, "Question and response must not be empty.")
    return await _run_locked(_run_grade, req)


@app.post("/attribution", response_model=AttributionResponse)
async def attribution(req: AttributionRequest, _: Annotated[None, Depends(require_model)]):
    return await _run_locked(_run_attribution, req)


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest, _: Annotated[None, Depends(require_model)]):
    return await _run_locked(_run_feedback, req)


@app.post("/correct")
async def correct(req: CorrectionRequest):
    _log({
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "event":            "correction",
        "question":         req.question,
        "response":         req.response,
        "max_grade":        req.max_grade,
        "predicted_grade":  req.predicted_grade,
        "human_grade":      req.human_grade,
        "comment":          req.comment,
    })
    return {"status": "ok"}


# ── Static ────────────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")
