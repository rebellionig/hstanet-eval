# syntax=docker/dockerfile:1
# Multi-stage build: "test" stage runs the test-suite, "runtime" stage is the slim image to ship.

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# ---- test stage: dev dependencies + tests ----
FROM base AS test
RUN pip install -e ".[dev]"
COPY tests ./tests
COPY examples ./examples
CMD ["pytest", "--cov=hstanet_eval", "--cov-fail-under=90"]

# ---- runtime stage: only the package and example data ----
FROM base AS runtime
RUN pip install . && useradd --create-home --uid 10001 app
COPY examples ./examples
USER app
ENTRYPOINT ["hstanet-eval"]
CMD ["--help"]
