"""Command-line interface: ``hstanet-eval metrics`` and ``hstanet-eval latency``."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .io import load_ground_truth, load_predictions
from .latency import benchmark
from .metrics import evaluate


def _cmd_metrics(args: argparse.Namespace) -> int:
    preds = load_predictions(args.pred)
    gts = load_ground_truth(args.gt)
    result = evaluate(preds, gts, iou_thresholds=args.iou, score_threshold=args.score_thr)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _resolve_target(spec: str):
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit("--target must look like 'package.module:function'")
    return getattr(importlib.import_module(module_name), attr)


def _cmd_latency(args: argparse.Namespace) -> int:
    fn = _resolve_target(args.target)
    shape = tuple(int(x) for x in args.shape.split(","))
    rng = np.random.default_rng(args.seed)
    inputs = (rng.random(shape, dtype=np.float32) for _ in range(args.warmup + args.runs))
    stats = benchmark(fn, inputs, warmup=args.warmup)
    print(json.dumps(stats, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hstanet-eval",
        description="Detection metrics and latency benchmarking for HSTANet experiments.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("metrics", help="compute AP / mAP / precision / recall / F1")
    m.add_argument("--pred", required=True, help="predictions JSON file")
    m.add_argument("--gt", required=True, help="ground-truth JSON file")
    m.add_argument(
        "--iou", type=float, nargs="+", default=[0.5], help="IoU thresholds (default 0.5)"
    )
    m.add_argument("--score-thr", type=float, default=0.0, help="confidence cut-off for P/R/F1")
    m.set_defaults(func=_cmd_metrics)

    lat = sub.add_parser("latency", help="benchmark inference latency of a callable")
    lat.add_argument("--target", required=True, help="callable as 'package.module:function'")
    lat.add_argument(
        "--shape", default="4,3,640,640", help="input shape, comma separated (T,C,H,W)"
    )
    lat.add_argument("--warmup", type=int, default=5)
    lat.add_argument("--runs", type=int, default=50)
    lat.add_argument("--seed", type=int, default=0)
    lat.set_defaults(func=_cmd_latency)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
