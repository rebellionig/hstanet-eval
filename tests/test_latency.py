import time

import pytest

from hstanet_eval import benchmark


def test_benchmark_reports_expected_keys_and_counts():
    stats = benchmark(lambda x: x, range(15), warmup=5)
    assert stats["n"] == 10
    for key in ("mean_ms", "std_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "fps"):
        assert key in stats
    assert stats["min_ms"] <= stats["p50_ms"] <= stats["p95_ms"] <= stats["max_ms"]


def test_benchmark_measures_real_time():
    stats = benchmark(lambda _: time.sleep(0.01), range(8), warmup=2)
    assert stats["mean_ms"] >= 10.0
    assert stats["fps"] <= 100.0


def test_warmup_calls_are_executed_but_not_timed():
    calls = []
    benchmark(calls.append, range(6), warmup=4)
    assert len(calls) == 6


def test_sync_hook_is_called_every_iteration():
    syncs = []
    benchmark(lambda x: x, range(7), warmup=2, sync=lambda: syncs.append(1))
    assert len(syncs) == 7


def test_too_few_inputs_raises():
    with pytest.raises(ValueError):
        benchmark(lambda x: x, range(3), warmup=5)
