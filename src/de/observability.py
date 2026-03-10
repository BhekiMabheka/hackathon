"""
Pipeline observability — metrics and health checks for the DE track.

Exposes Prometheus metrics (counters, histograms, gauges) that can be
scraped by a monitoring stack (Prometheus + Grafana).

In a competition context, use the lightweight TextFileCollector:
  - Metrics are written to a .prom file.
  - No server needed locally; a CI job can archive them.

Key metrics:
  - pipeline_run_total          — counter, labelled by stage / status
  - pipeline_rows_processed     — gauge, rows per pipeline stage
  - pipeline_null_rate          — gauge, null rate per column (drift signal)
  - pipeline_duration_seconds   — histogram, stage processing time
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable

from src.utils.logging import get_logger

log = get_logger(__name__)

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        CollectorRegistry,
        write_to_textfile,
        REGISTRY,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    log.warning("prometheus_client not installed — observability metrics disabled")


# ---------------------------------------------------------------------------
# Metric definitions (only created if prometheus_client is available)
# ---------------------------------------------------------------------------

if _PROMETHEUS_AVAILABLE:
    PIPELINE_RUNS = Counter(
        "pipeline_run_total",
        "Total pipeline runs",
        ["stage", "status"],
    )
    ROWS_PROCESSED = Gauge(
        "pipeline_rows_processed",
        "Rows processed per stage",
        ["stage"],
    )
    NULL_RATE = Gauge(
        "pipeline_null_rate",
        "Null rate per column",
        ["stage", "column"],
    )
    DURATION = Histogram(
        "pipeline_duration_seconds",
        "Stage execution time in seconds",
        ["stage"],
        buckets=[1, 5, 10, 30, 60, 120, 300, 600],
    )


@contextmanager
def track_stage(stage: str, output_file: str | None = None):
    """
    Context manager that tracks duration and success/failure of a pipeline stage.

    Usage
    -----
    >>> with track_stage("preprocessing"):
    ...     result = run_preprocessing()
    """
    if not _PROMETHEUS_AVAILABLE:
        yield
        return

    start = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - start
        PIPELINE_RUNS.labels(stage=stage, status="success").inc()
        DURATION.labels(stage=stage).observe(elapsed)
        log.info("Stage complete", stage=stage, duration_s=round(elapsed, 2))
    except Exception as exc:
        PIPELINE_RUNS.labels(stage=stage, status="failure").inc()
        log.error("Stage failed", stage=stage, error=str(exc))
        raise
    finally:
        if output_file:
            _write_metrics(output_file)


def record_dataframe_metrics(stage: str, df, date_col: str | None = None) -> None:
    """
    Record row count and null rates for a DataFrame after a pipeline stage.
    """
    if not _PROMETHEUS_AVAILABLE:
        return
    ROWS_PROCESSED.labels(stage=stage).set(len(df))
    for col in df.columns:
        null_rate = float(df[col].isnull().mean())
        NULL_RATE.labels(stage=stage, column=col).set(null_rate)
    log.info("Recorded dataframe metrics", stage=stage, rows=len(df))


def _write_metrics(output_file: str) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    write_to_textfile(output_file, REGISTRY)
    log.info("Metrics written", path=output_file)


def timed(stage: str):
    """Decorator version of track_stage for functions."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with track_stage(stage):
                return fn(*args, **kwargs)
        return wrapper
    return decorator
