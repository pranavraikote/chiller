"""Feedback model in a killable subprocess.

bitsandbytes-quantized models don't release their VRAM on in-process offload (the quantized weight
buffers live outside PyTorch's caching allocator, so `empty_cache()` can't reclaim them). The only
reliable way to fully free a quantized 14B on idle is to run it in a child process and **kill** the
process — which tears down the CUDA context and returns all memory to the driver.

The parent picks the GPU (avoiding the grader's card) and retries on OOM; the child just loads on the
GPU it's told and serves greedy generations over a pipe. Singleton: fb_mgr.
"""

import multiprocessing as mp
import threading
import time

from config import FEEDBACK_8BIT, FEEDBACK_GPU, FEEDBACK_MODEL, IDLE_TIMEOUT, RESERVED_GPUS
from gpu import _pick_gpu
from logging_setup import log

_ctx = mp.get_context("spawn")   # fork after CUDA init is unsafe — spawn a fresh interpreter


def _worker(conn, gpu: int, model_name: str, use_8bit: bool):
    """Child process: load the model on `gpu`, then serve {system,user,max_new_tokens} → {text}."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        quant = BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, quantization_config=quant,
            device_map={"": gpu}, trust_remote_code=True,
        ).eval()
        try:
            torch.cuda.set_device(gpu)   # keep this process's default off GPU 0
        except Exception:
            pass
        conn.send({"ready": gpu})
    except Exception as e:  # report OOM distinctly so the parent can re-pick a card
        conn.send({"oom" if "out of memory" in str(e).lower() else "error": str(e)})
        return

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        try:
            msgs = [{"role": "system", "content": msg["system"]},
                    {"role": "user",   "content": msg["user"]}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=msg["max_new_tokens"], do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            conn.send({"text": tok.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()})
        except Exception as e:
            conn.send({"error": str(e)})


class FeedbackProcess:
    """Parent-side manager: spawns/kills the feedback worker and relays generation requests.

    Mirrors the lazy-load / idle-offload interface of the in-process managers, but "offload" means
    killing the child process so the quantized model's VRAM is actually released.
    """

    _label = "Feedback model"

    def __init__(self):
        self.proc = None
        self.conn = None
        self.gpu  = None
        self.lock = threading.Lock()
        self.last_used = 0.0
        self._watchdog = None

    @property
    def available(self):
        return bool(FEEDBACK_MODEL)

    @property
    def loaded(self):
        return self.proc is not None and self.proc.is_alive()

    def _spawn(self, exclude_gpu, attempts: int = 6):
        """Start the worker on the most-free card (excluding the grader's), retrying on OOM."""
        exclude = set()
        if exclude_gpu is not None:
            exclude.add(exclude_gpu)
        last = None
        for attempt in range(attempts):
            # Pin to FEEDBACK_GPU; fall back to the most-free card only on OOM.
            if attempt == 0 and FEEDBACK_GPU not in exclude and FEEDBACK_GPU not in RESERVED_GPUS:
                gpu = FEEDBACK_GPU
            else:
                gpu = _pick_gpu(exclude)
            parent_conn, child_conn = _ctx.Pipe()
            proc = _ctx.Process(target=_worker, args=(child_conn, gpu, FEEDBACK_MODEL, FEEDBACK_8BIT),
                                daemon=True)
            proc.start()
            child_conn.close()
            if not parent_conn.poll(timeout=600):
                proc.terminate(); proc.join(timeout=10)
                last = "worker load timed out"
                continue
            msg = parent_conn.recv()
            if "ready" in msg:
                self.proc, self.conn, self.gpu = proc, parent_conn, msg["ready"]
                self.last_used = time.time()
                log.info(f"Feedback worker ready (pid {proc.pid}) → cuda:{self.gpu}")
                return
            proc.terminate(); proc.join(timeout=10)
            last = msg.get("oom") or msg.get("error")
            if "oom" in msg:
                log.warning(f"Feedback worker OOM on cuda:{gpu} — excluding and retrying.")
                exclude.add(gpu)
                continue
            break   # non-OOM load error — don't thrash
        raise RuntimeError(f"Feedback worker could not load after {attempts} attempts: {last}")

    def generate(self, system: str, user: str, max_new_tokens: int = 220, exclude_gpu=None) -> str:
        with self.lock:
            if not self.loaded:
                self._spawn(exclude_gpu)
                self._start_watchdog()
            try:
                self.conn.send({"system": system, "user": user, "max_new_tokens": max_new_tokens})
                if not self.conn.poll(timeout=300):
                    raise RuntimeError("feedback worker generation timed out")
                msg = self.conn.recv()
            except (EOFError, BrokenPipeError, OSError) as e:
                self._kill()
                raise RuntimeError(f"feedback worker died: {e}")
            self.last_used = time.time()
            if "error" in msg:
                raise RuntimeError(msg["error"])
            return msg["text"]

    def offload(self):
        with self.lock:
            self._kill()

    def _kill(self):
        if self.proc is not None:
            log.info(f"{self._label} offloaded (killing worker pid {self.proc.pid}).")
            try:
                self.conn.send(None)
            except Exception:
                pass
            self.proc.terminate()
            self.proc.join(timeout=10)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join()
        self.proc = None
        self.conn = None
        self.gpu  = None

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
                        log.info(f"{self._label} idle timeout — killing worker.")
                        self._kill()
                        return

        self._watchdog = threading.Thread(target=_watch, daemon=True)
        self._watchdog.start()


fb_mgr = FeedbackProcess()
