# hstanet-eval

   ## Docker and CI
   Built and tested automatically by Jenkins; image: armansalken/hstanet-eval

[![CI](https://github.com/rebellionig/hstanet-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/rebellionig/hstanet-eval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Small, dependency-light toolkit for evaluating **real-time video object detection**
experiments: detection metrics (IoU, precision, recall, F1, AP, mAP) and an inference
latency benchmark.

It was written as a supporting tool for **HSTANet** (Hybrid Spatio-Temporal Attention
Network), a master's research project at Astana IT University, and closes two gaps found
during its architecture review: *latency of inference was never measured* (NFR1) and
*evaluation was not reproducible outside the training script* (NFR3).

## Features

- `iou_matrix` — vectorised pairwise IoU.
- `evaluate` — AP@IoU per class, mAP over several IoU thresholds, micro-averaged
  precision / recall / F1 at a chosen confidence cut-off. Greedy score-ordered matching and
  all-point AP interpolation (PASCAL VOC style).
- `benchmark` — warm-up aware latency measurement: mean, std, p50, p95, FPS; optional
  `sync` hook for asynchronous GPU work (e.g. `torch.cuda.synchronize`).
- A CLI for both, so results can be produced in CI or from shell scripts.

## Installation

```bash
git clone https://github.com/rebellionig/hstanet-eval.git
cd hstanet-eval
pip install -e .          # runtime: numpy only
pip install -e ".[dev]"   # + pytest, coverage, ruff, build
```

Requires Python 3.10 or newer.

## Usage

### Metrics

```bash
hstanet-eval metrics \
  --pred examples/predictions.json \
  --gt examples/ground_truth.json \
  --iou 0.5 0.75 --score-thr 0.5
```

Python API:

```python
from hstanet_eval import ImageDetections, ImageGroundTruth, evaluate

preds = {"img1": ImageDetections([[0, 0, 10, 10]], [0.9], [0])}
gts = {"img1": ImageGroundTruth([[0, 0, 10, 9]], [0])}
result = evaluate(preds, gts, iou_thresholds=(0.5, 0.75))
print(result.map, result.f1)
```

Input files are JSON mappings from image id to boxes in `[x1, y1, x2, y2]` format
(see `examples/`). Predictions need `boxes`, `scores`, `labels`; ground truth needs
`boxes`, `labels`.

### Latency

```bash
hstanet-eval latency --target hstanet_eval.demo:dummy_model --shape 4,3,640,640 --runs 50
```

`--target` is any importable callable (`package.module:function`) that accepts a NumPy
array — typically a thin wrapper around your model's forward pass. The shape is
`(T, C, H, W)` for a temporal window.

## Technology choices

| Choice | Why |
|---|---|
| Python ≥ 3.10 | Same language as the research code (PyTorch / YOLOv8); fast iteration. |
| NumPy only | Metrics are array maths; no heavy ML dependency, so CI installs in seconds and the tool runs anywhere. |
| `src/` layout + `pyproject.toml` | Tests run against the installed package; standard, tool-agnostic packaging. |
| pytest + pytest-cov | De-facto standard; hand-computed test cases double as metric documentation. |
| ruff | One fast tool for linting, import sorting and formatting. |
| GitHub Actions | CI lives next to the code; matrix over Python 3.10–3.13. |

## Development

```bash
pytest --cov=hstanet_eval
ruff check . && ruff format --check .
```

CI (`.github/workflows/ci.yml`) runs lint, the test matrix with a 90% coverage gate, a CLI
smoke test and a package build on every push and pull request. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Project layout

```
src/hstanet_eval/   boxes.py  metrics.py  latency.py  io.py  cli.py  demo.py
tests/              unit tests for every module
examples/           sample predictions / ground truth
.github/            CI workflow, issue and PR templates, Dependabot
```

## License

[MIT](LICENSE)
