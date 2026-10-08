"""Inference latency benchmark.

Measures wall-clock time per call of an arbitrary callable (e.g. a model's
forward pass), after a warm-up phase, and reports mean / percentile latency
and throughput. Targets NFR1 (near-real-time inference) of the HSTANet project.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


def benchmark(
    fn: Callable[[Any], Any],
    inputs: Iterable[Any],
    warmup: int = 5,
    sync: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Time ``fn`` on each input.

    Args:
        fn: callable taking one input (a frame or a temporal window of frames).
        inputs: iterable of inputs. The first ``warmup`` are run but not timed.
        warmup: number of untimed warm-up calls (GPU init, caches, JIT).
        sync: optional barrier called after every ``fn`` call, e.g.
            ``torch.cuda.synchronize`` so asynchronous GPU work is included.

    Returns:
        Dict with ``n``, ``mean_ms``, ``std_ms``, ``p50_ms``, ``p95_ms``,
        ``min_ms``, ``max_ms`` and ``fps``.
    """
    samples: list[float] = []
    for i, item in enumerate(inputs):
        start = time.perf_counter()
        fn(item)
        if sync is not None:
            sync()
        elapsed = time.perf_counter() - start
        if i >= warmup:
            samples.append(elapsed * 1000.0)

    if not samples:
        raise ValueError("no timed samples: provide more inputs than warmup calls")

    arr = np.asarray(samples)
    mean = float(arr.mean())
    return {
        "n": int(arr.size),
        "mean_ms": mean,
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "fps": 1000.0 / mean if mean > 0 else float("inf"),
    }
