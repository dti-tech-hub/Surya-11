"""Computational-resource measurement: time, memory, CPU, model size.

Measurements use real instrumentation (perf_counter, tracemalloc, psutil).
Where a metric is unreliable on the current machine it is reported as NaN with
a documented limitation instead of a fabricated number.
"""
from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np
import psutil


@contextmanager
def stopwatch():
    """Context manager yielding elapsed wall-clock seconds."""
    t0 = time.perf_counter()
    result = {"seconds": 0.0}
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - t0


@contextmanager
def measure_peak_memory_mb():
    """Peak *Python-allocated* memory during the block, in MB (tracemalloc)."""
    tracemalloc.start()
    result = {"peak_mb": float("nan")}
    try:
        yield result
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["peak_mb"] = peak / (1024**2)


def measure_cpu_percent(fn: Callable[[], Any], interval: float = 0.05) -> tuple[Any, float]:
    """Run ``fn`` while sampling process CPU utilisation; returns (result, cpu%)."""
    proc = psutil.Process()
    proc.cpu_percent(None)  # prime
    result = fn()
    time.sleep(interval)  # allow the kernel counters to advance
    cpu = proc.cpu_percent(None)
    return result, float(cpu)


def measure_inference_time(predict_fn: Callable[[], np.ndarray], repeats: int = 3) -> dict[str, Any]:
    """Wall time and per-sample latency of an inference call."""
    durations: list[float] = []
    n_samples = 0
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = predict_fn()
        durations.append(time.perf_counter() - t0)
        n_samples = int(np.asarray(out).shape[0])
    best = min(durations)
    return {
        "seconds_mean": float(np.mean(durations)),
        "seconds_best": float(best),
        "per_sample_ms": float(1000.0 * best / max(n_samples, 1)),
        "repeats": max(1, repeats),
        "n_samples": n_samples,
    }


def model_resource_report(model: Any, X: np.ndarray | None = None, predict_fn: Callable[[], np.ndarray] | None = None) -> dict[str, Any]:
    """Parameters, size, inference time, peak memory for a fitted model."""
    report: dict[str, Any] = {
        "model_name": getattr(model, "name", type(model).__name__),
        "n_parameters": int(getattr(model, "n_params", 0) or 0),
        "model_size_kb": float(getattr(model, "size_bytes", lambda: 0)() / 1024),
    }
    if predict_fn is None and X is not None and hasattr(model, "predict_proba"):
        predict_fn = lambda: model.predict_proba(X)  # noqa: E731
    if predict_fn is not None:
        with measure_peak_memory_mb() as mem:
            report["inference"] = measure_inference_time(predict_fn, repeats=3)
        report["inference_peak_memory_mb"] = mem["peak_mb"]
    else:
        report["inference"] = None
        report["inference_peak_memory_mb"] = float("nan")
    return report


def rss_memory_mb() -> float:
    """Current resident-set size of this process (MB)."""
    return psutil.Process().memory_info().rss / (1024**2)


def measure_call_rss(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run fn; return (result, RSS growth in MB across the call)."""
    proc = psutil.Process()
    before = proc.memory_info().rss
    result = fn()
    after = proc.memory_info().rss
    return result, (after - before) / (1024**2)
