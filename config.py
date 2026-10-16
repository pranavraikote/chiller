"""Configuration — environment knobs and one-time GPU visibility setup.

Imported first (before torch) so the CUDA_VISIBLE_DEVICES selection below takes effect.
Every path below has a neutral default and can be overridden with an environment variable.
"""

import os
from pathlib import Path

# Single-GPU dev runs: pick the most-free card up front. In two-model mode (FEEDBACK_MODEL set) we
# leave all cards visible and each model picks its GPU at LOAD time instead (see gpu._pick_gpu) —
# deploy-time free memory is stale by the time the lazily-loaded model needs the card on a shared cluster.
if "CUDA_VISIBLE_DEVICES" not in os.environ and not os.environ.get("FEEDBACK_MODEL"):
    try:
        import pynvml
        pynvml.nvmlInit()
        best, best_free = 0, 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            free = int(pynvml.nvmlDeviceGetMemoryInfo(h).free)
            if free > best_free:
                best_free, best = free, i
        pynvml.nvmlShutdown()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best)
    except Exception:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Model ──────────────────────────────────────────────────────────────────────
# BASE_MODEL is pulled from the Hugging Face Hub. ADAPTER_DIR points at a local LoRA adapter
# directory (see README for how to obtain/train one); default is ./adapter next to the server.
BASE_MODEL  = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "adapter")

# ── Server ─────────────────────────────────────────────────────────────────────
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT", "300"))
LOG_FILE     = os.environ.get("LOG_FILE", "logs/grades.jsonl")
LOG_DIR      = Path(LOG_FILE).parent
MAX_FIELD    = int(os.environ.get("MAX_FIELD_CHARS", "3000"))

# ── RAG grounded feedback ───────────────────────────────────────────────────────
# RAG_DIR must hold chunks.faiss_index + chunks.json (a lecture/reference corpus). Default ./rag.
RAG_DIR       = os.environ.get("RAG_DIR", "rag")
RAG_INDEX     = os.environ.get("RAG_INDEX",  f"{RAG_DIR}/chunks.faiss_index")
RAG_CHUNKS    = os.environ.get("RAG_CHUNKS", f"{RAG_DIR}/chunks.json")
EMBED_MODEL   = os.environ.get("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
USE_RAG       = os.environ.get("USE_RAG", "1") != "0"
RAG_TOP_K     = int(os.environ.get("RAG_TOP_K", "5"))
RAG_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.35"))  # abstain below → ungrounded fallback

# ── Dedicated feedback model (optional) ──────────────────────────────────────────
# A larger instruct model on its OWN GPU gives more faithful prose feedback; grading stays on the
# 7B+LoRA. Leave FEEDBACK_MODEL empty to run everything on one GPU: feedback then falls back to the
# grader's base model (adapter disabled). Set e.g. FEEDBACK_MODEL=Qwen/Qwen2.5-14B-Instruct to enable.
FEEDBACK_MODEL = os.environ.get("FEEDBACK_MODEL", "")
FEEDBACK_8BIT  = os.environ.get("FEEDBACK_8BIT", "1") != "0"

# GPUs the picker must never use (reserved for other jobs). Default: none.
RESERVED_GPUS = {int(x) for x in os.environ.get("RESERVED_GPUS", "").split(",") if x.strip() != ""}

# Pinned cards (container-local cuda index; CUDA_DEVICE_ORDER=PCI_BUS_ID makes these the physical
# indices). Grader on GRADER_GPU, feedback worker on FEEDBACK_GPU. The dynamic picker is only a
# fallback if a pinned card can't fit at load time.
GRADER_GPU   = int(os.environ.get("GRADER_GPU", "0"))
FEEDBACK_GPU = int(os.environ.get("FEEDBACK_GPU", "1"))
