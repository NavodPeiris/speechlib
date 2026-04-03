"""
Step-level wall-time and VRAM profiler.

Activate with:
    SPEECHLIB_PROFILE=1 python process_audio.py

Reports elapsed time per pipeline step plus GPU VRAM before/after for GPU steps.
Zero overhead when the env var is unset.

API:
    @timed(step_name)        — decorator for functions
    measure(step_name, gpu)  — context manager for inline blocks
    print_report()           — tabular summary at end of pipeline
    reset()                  — clear measurements (useful in tests)
"""
import os
import time
from contextlib import contextmanager
from functools import wraps

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_GPU_STEPS = {"enhance_audio", "diarization", "transcription"}

_records: list[dict] = []


def _enabled() -> bool:
    return os.environ.get("SPEECHLIB_PROFILE", "0") not in ("0", "", "false", "False")


def _vram_mb() -> float | None:
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return None


def _sync_gpu() -> None:
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.synchronize()


def _record(step: str, elapsed: float, mem_before: float | None, mem_after: float | None) -> None:
    _records.append({
        "step": step,
        "elapsed": elapsed,
        "mem_before": mem_before,
        "mem_after": mem_after,
    })


def timed(step_name: str):
    """Decorator — records wall time (and VRAM for GPU steps) of a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _enabled():
                return func(*args, **kwargs)

            is_gpu = step_name in _GPU_STEPS
            if is_gpu:
                _sync_gpu()
            mem_before = _vram_mb() if is_gpu else None

            t0 = time.perf_counter()
            result = func(*args, **kwargs)

            if is_gpu:
                _sync_gpu()
            elapsed = time.perf_counter() - t0

            mem_after = _vram_mb() if is_gpu else None
            _record(step_name, elapsed, mem_before, mem_after)
            return result
        return wrapper
    return decorator


@contextmanager
def measure(step_name: str, gpu: bool = False):
    """Context manager — records wall time of an inline block."""
    if not _enabled():
        yield
        return

    if gpu:
        _sync_gpu()
    mem_before = _vram_mb() if gpu else None

    t0 = time.perf_counter()
    yield

    if gpu:
        _sync_gpu()
    elapsed = time.perf_counter() - t0

    mem_after = _vram_mb() if gpu else None
    _record(step_name, elapsed, mem_before, mem_after)


def reset() -> None:
    """Clear all measurements (useful between pipeline runs in tests)."""
    _records.clear()


def print_report() -> None:
    """Print tabular timing report. No-op when profiling is disabled."""
    if not _enabled() or not _records:
        return

    col_step = 24
    col_time = 10
    col_mem  = 13

    header = (
        f"{'Step':<{col_step}}  {'Time':>{col_time}}  "
        f"{'VRAM before':>{col_mem}}  {'VRAM after':>{col_mem}}  {'VRAM delta':>{col_mem}}"
    )
    sep = "-" * len(header)

    print()
    print("=== Step Timer Report ===")
    print()
    print(header)
    print(sep)

    total = 0.0
    for r in _records:
        elapsed = r["elapsed"]
        total  += elapsed
        mb_b    = r["mem_before"]
        mb_a    = r["mem_after"]

        t_str = f"{elapsed:.3f}s"
        b_str = f"{mb_b:.0f} MB" if mb_b is not None else "-"
        a_str = f"{mb_a:.0f} MB" if mb_a is not None else "-"
        d_str = f"{mb_a - mb_b:+.0f} MB" if mb_b is not None and mb_a is not None else "-"

        print(
            f"{r['step']:<{col_step}}  {t_str:>{col_time}}  "
            f"{b_str:>{col_mem}}  {a_str:>{col_mem}}  {d_str:>{col_mem}}"
        )

    print(sep)
    print(f"{'TOTAL':<{col_step}}  {total:.3f}s")
    print()
