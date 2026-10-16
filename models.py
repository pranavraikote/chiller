"""Grader model manager — Qwen2.5-7B + LoRA (bf16), lazy-load with idle auto-offload.

The dedicated feedback model lives in feedback_proc.py (a killable subprocess), because bitsandbytes
can't release quantized VRAM in-process. Singleton here: mgr (the grader).
"""

import gc
import threading
import time
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import (
    ADAPTER_DIR, BASE_MODEL, EMBED_MODEL, FEEDBACK_GPU, FEEDBACK_MODEL,
    GRADER_GPU, IDLE_TIMEOUT, RAG_CHUNKS, RAG_INDEX, USE_RAG,
)
from gpu import _load_balanced
from logging_setup import log
from rag import LectureRetriever


class _LazyModel:
    """Lazy GPU loader with idle auto-offload. Subclasses implement _load().

    Holds the shared machinery for both models: a load lock, the chosen GPU, an idle watchdog that
    offloads after IDLE_TIMEOUT, and cleanup. `_label` names the instance in the logs.
    """

    _label = "Model"

    def __init__(self):
        self.model:     Any = None
        self.tokenizer: Any = None
        self.gpu:       int = 0             # cuda index chosen at load time
        self.lock      = threading.Lock()
        self.last_used = 0.0
        self._watchdog = None

    @property
    def loaded(self):
        return self.model is not None

    def ensure_loaded(self):
        with self.lock:
            if self.loaded:
                self.last_used = time.time()
                return
            self._load()
            self.last_used = time.time()
            self._start_watchdog()

    def _load(self):
        raise NotImplementedError

    def offload(self):
        with self.lock:
            if not self.loaded:
                return
            self._cleanup()
            log.info(f"{self._label} offloaded.")

    def _cleanup(self):
        self.model     = None
        self.tokenizer = None
        gc.collect()
        # Scope to this model's card: empty_cache() with no device targets the CALLING thread's
        # current device (default 0), and offload runs in the watchdog / executor threads — so a bare
        # call would strand a ~270 MB context on GPU 0. torch.cuda.device() is thread-safe.
        with torch.cuda.device(self.gpu):
            torch.cuda.empty_cache()

    def touch(self):
        self.last_used = time.time()

    def _start_watchdog(self):
        if self._watchdog and self._watchdog.is_alive():
            return

        def _watch():
            while True:
                time.sleep(30)
                with self.lock:
                    if not self.loaded:
                        return
                    if time.time() - self.last_used >= IDLE_TIMEOUT:
                        log.info(f"{self._label} idle timeout — offloading.")
                        self._cleanup()
                        return

        self._watchdog = threading.Thread(target=_watch, daemon=True)
        self._watchdog.start()


class ModelManager(_LazyModel):
    """Grader — 7B + LoRA (bf16). Adds the resident (CPU) lecture retriever for grounded feedback."""

    _label = "Grader"

    def __init__(self):
        super().__init__()
        self.gpu = GRADER_GPU               # so even a pre-load cleanup never targets GPU 0
        self.retriever: Any = None          # resident (CPU) — not offloaded with the model
        self._rag_lock = threading.Lock()

    def ensure_retriever(self):
        """Lazy-load the lecture retriever once; kept resident across model offloads.

        On failure, leaves retriever=None so feedback degrades gracefully to ungrounded.
        """
        if not USE_RAG or self.retriever is not None:
            return
        with self._rag_lock:
            if self.retriever is not None:
                return
            log.info("Loading lecture retriever (FAISS + embedder, CPU) …")
            try:
                self.retriever = LectureRetriever(RAG_INDEX, RAG_CHUNKS, EMBED_MODEL, device="cpu")
                log.info(f"Retriever ready: {len(self.retriever.chunks)} chunks.")
            except Exception as e:
                log.error(f"Retriever load failed ({type(e).__name__}): {e} — feedback stays ungrounded.")
                self.retriever = None

    def _load(self):
        log.info("Loading tokenizer …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log.info("Loading model to GPU (bfloat16) …")
        try:
            # Pin the grader to a single card (chosen now, by free memory) when a dedicated feedback
            # model is in play, so accelerate can't shard it across GPUs. Single-GPU dev runs keep "auto".
            def _build_grader(gpu):
                dm = {"": gpu} if FEEDBACK_MODEL else "auto"
                base = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL, dtype=torch.bfloat16, device_map=dm, trust_remote_code=True,
                )
                return PeftModel.from_pretrained(base, ADAPTER_DIR)

            if FEEDBACK_MODEL:
                # Pin to GRADER_GPU; fall back to the most-free card (not the feedback card) only on OOM.
                self.model, self.gpu = _load_balanced(
                    _build_grader, exclude={FEEDBACK_GPU}, prefer=GRADER_GPU)
                log.info(f"Grader → cuda:{self.gpu}")
            else:
                self.model = _build_grader(0)
            self.model.eval()
            log.info("Model ready.")
        except Exception as e:
            log.error(f"Model load error ({type(e).__name__}): {e}")
            self._cleanup()
            raise RuntimeError(f"GPU unavailable: {e}")


mgr = ModelManager()
