# Contributing

## Workflow

1. Open or pick an [issue](https://github.com/rebellionig/hstanet-eval/issues).
2. Create a short-lived branch from `main`: `feature/<topic>` or `fix/<topic>`.
3. Make small commits with imperative messages (`Add AP@[.5:.95] helper`).
4. Open a pull request. CI must pass (lint, tests on Python 3.10–3.13, build) before merging.

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest --cov=hstanet_eval
ruff check . && ruff format --check .
```

## Conventions

- Every new metric needs a hand-computed test case (see `tests/test_metrics.py`).
- Public functions have docstrings with argument and return descriptions.
- Releases are tagged `vX.Y.Z` on `main`.
