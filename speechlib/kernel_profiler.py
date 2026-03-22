"""
CPU vs CUDA kernel profiler via torch.profiler.

Activate with:
    SPEECHLIB_PROFILE_KERNELS=1 python process_audio.py

For each instrumented GPU step, records how much time was spent in CPU operations
vs CUDA kernels. Exports a Chrome-compatible trace JSON per step.

API:
    @timed(step_name)        — decorator for GPU functions
    measure(step_name)       — context manager for GPU inline blocks
    print_report()           — CPU/CUDA breakdown table + trace file paths
    reset()                  — clear measurements

Traces are saved to ./profiling_traces/ and can be opened at:
    chrome://tracing
    https://ui.perfetto.dev
"""
import os
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

try:
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_TRACES_DIR = Path("profiling_traces")

_records: list[dict] = []


def _enabled() -> bool:
    return os.environ.get("SPEECHLIB_PROFILE_KERNELS", "0") not in ("0", "", "false", "False")


def _cuda_available() -> bool:
    return _TORCH_AVAILABLE and torch.cuda.is_available()


def _sync() -> None:
    if _cuda_available():
        torch.cuda.synchronize()


def _vram_mb() -> float | None:
    if _cuda_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return None


def _run_under_profiler(step_name: str, fn, args, kwargs):
    """Run fn(*args, **kwargs) wrapped in torch.profiler; record CPU/CUDA times."""
    _TRACES_DIR.mkdir(exist_ok=True)
    _sync()
    mem_before = _vram_mb()
    t0 = time.perf_counter()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=False,
    ) as prof:
        with record_function(step_name):
            result = fn(*args, **kwargs)

    _sync()
    elapsed = time.perf_counter() - t0
    mem_after = _vram_mb()

    avgs = prof.key_averages()
    cpu_ms  = sum(e.cpu_time_total  for e in avgs) / 1000   # µs → ms
    cuda_ms = sum(e.cuda_time_total for e in avgs) / 1000

    trace_path = _TRACES_DIR / f"{step_name}_trace.json"
    prof.export_chrome_trace(str(trace_path))

    _records.append({
        "step":       step_name,
        "elapsed":    elapsed,
        "mem_before": mem_before,
        "mem_after":  mem_after,
        "cpu_ms":     cpu_ms,
        "cuda_ms":    cuda_ms,
        "trace":      str(trace_path),
    })
    return result


def timed(step_name: str):
    """Decorator — profiles CPU/CUDA kernel time for a GPU function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _enabled() or not _cuda_available():
                return func(*args, **kwargs)
            return _run_under_profiler(step_name, func, args, kwargs)
        return wrapper
    return decorator


@contextmanager
def measure(step_name: str):
    """Context manager — profiles CPU/CUDA kernel time for an inline block."""
    if not _enabled() or not _cuda_available():
        yield
        return

    _TRACES_DIR.mkdir(exist_ok=True)
    _sync()
    mem_before = _vram_mb()
    t0 = time.perf_counter()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=False,
    ) as prof:
        with record_function(step_name):
            yield

    _sync()
    elapsed = time.perf_counter() - t0
    mem_after = _vram_mb()

    avgs = prof.key_averages()
    cpu_ms  = sum(e.cpu_time_total  for e in avgs) / 1000
    cuda_ms = sum(e.cuda_time_total for e in avgs) / 1000

    trace_path = _TRACES_DIR / f"{step_name}_trace.json"
    prof.export_chrome_trace(str(trace_path))

    _records.append({
        "step":       step_name,
        "elapsed":    elapsed,
        "mem_before": mem_before,
        "mem_after":  mem_after,
        "cpu_ms":     cpu_ms,
        "cuda_ms":    cuda_ms,
        "trace":      str(trace_path),
    })


def reset() -> None:
    """Clear all measurements."""
    _records.clear()


def print_report() -> None:
    """Print CPU/CUDA breakdown table and trace file paths."""
    if not _enabled() or not _records:
        return

    col_step = 24
    col_ms   = 12
    col_pct  =  7

    header = (
        f"{'Step':<{col_step}}  {'CPU time':>{col_ms}}  {'CUDA time':>{col_ms}}  "
        f"{'CPU %':>{col_pct}}  {'CUDA %':>{col_pct}}  {'VRAM delta':>12}"
    )
    sep = "-" * len(header)

    print()
    print("=== Kernel Profiler Report (torch.profiler) ===")
    print()
    print(header)
    print(sep)

    for r in _records:
        cpu  = r["cpu_ms"]
        cuda = r["cuda_ms"]
        total_op = cpu + cuda
        cpu_pct  = f"{100 * cpu  / total_op:.1f}%" if total_op > 0 else "-"
        cuda_pct = f"{100 * cuda / total_op:.1f}%" if total_op > 0 else "-"
        mb_b = r["mem_before"]
        mb_a = r["mem_after"]
        d_str = f"{mb_a - mb_b:+.0f} MB" if mb_b is not None and mb_a is not None else "-"

        print(
            f"{r['step']:<{col_step}}  {cpu:>{col_ms}.1f} ms  {cuda:>{col_ms}.1f} ms  "
            f"{cpu_pct:>{col_pct}}  {cuda_pct:>{col_pct}}  {d_str:>12}"
        )

    print()
    print(f"Traces saved to {_TRACES_DIR}/")
    for r in _records:
        print(f"  {r['trace']}")
    print("Open at: chrome://tracing  or  https://ui.perfetto.dev")
    print()
