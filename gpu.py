"""GPU selection for a shared cluster: pick the most-free card at load time, retry on OOM."""

import gc

import torch

from config import RESERVED_GPUS
from logging_setup import log


def _fallback_gpu(exclude: set[int]) -> int:
    """Lowest allowed cuda index when free-memory info is unavailable (never a reserved card)."""
    try:
        n = torch.cuda.device_count()
    except Exception:
        n = 1
    return next((i for i in range(max(n, 1)) if i not in exclude), 0)


def _pick_gpu(exclude: set[int] | None = None) -> int:
    """cuda index of the most-free visible GPU, chosen at LOAD time (not deploy time).

    Never returns a RESERVED_GPUS card (reserved for other jobs) and skips the `exclude` set (e.g. the
    grader's card) so the two models land on different GPUs. Requires CUDA_DEVICE_ORDER=PCI_BUS_ID so
    torch's cuda:i matches NVML index i.
    """
    exclude = set(exclude or set()) | RESERVED_GPUS
    try:
        import pynvml
        pynvml.nvmlInit()
        best, best_free = None, -1
        for i in range(pynvml.nvmlDeviceGetCount()):
            if i in exclude:
                continue
            free = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).free)
            if free > best_free:
                best_free, best = free, i
        pynvml.nvmlShutdown()
        return best if best is not None else _fallback_gpu(exclude)
    except Exception:
        return _fallback_gpu(exclude)


def _is_oom(e: Exception) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
    )


def _load_balanced(build, exclude=None, prefer=None, attempts: int = 6):
    """Load a model on `prefer` (if given and allowed), else the most-free GPU, re-picking on OOM.

    Normal path is the pinned `prefer` card; the dynamic most-free pick is only a fallback if the
    pinned card OOMs (e.g. transient contention), never a reserved card. Returns (model, gpu_index).
    """
    exclude = set(exclude or set()) | RESERVED_GPUS
    last = None
    for attempt in range(attempts):
        if attempt == 0 and prefer is not None and prefer not in exclude:
            gpu = prefer
        else:
            gpu = _pick_gpu(exclude)
        try:
            return build(gpu), gpu
        except Exception as e:  # noqa: BLE001 — re-pick only on OOM, re-raise anything else
            if not _is_oom(e):
                raise
            last = e
            log.warning(f"OOM loading on cuda:{gpu} — excluding it and retrying.")
            exclude.add(gpu)
            gc.collect()
            with torch.cuda.device(gpu):   # scope to the card we tried, not the thread default (0)
                torch.cuda.empty_cache()
    raise RuntimeError(f"No GPU had enough free memory after {attempts} attempts: {last}")
