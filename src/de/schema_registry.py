"""
Schema registry and drift detection for the DE track.

In production banking pipelines, schema drift is one of the most common
silent failures: a source system adds a column, changes a type, or starts
sending nulls for a previously populated field — and the pipeline silently
degrades.

This module:
  1. Registers an expected schema (column names, types, nullable flags).
  2. Compares incoming data against the registered schema.
  3. Emits structured drift events that can feed alerts or circuit breakers.

Design: schemas are stored as JSON files so they can be versioned in git.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ColumnSchema:
    name: str
    dtype: str           # pandas dtype string (e.g., 'int64', 'object', 'float64')
    nullable: bool = True
    min_val: Any = None
    max_val: Any = None
    allowed_values: list | None = None  # for low-cardinality categoricals


@dataclass
class SchemaDefinition:
    name: str
    version: str
    columns: list[ColumnSchema] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SchemaDefinition":
        with open(path) as f:
            data = json.load(f)
        cols = [ColumnSchema(**c) for c in data.pop("columns")]
        return cls(**data, columns=cols)

    @classmethod
    def infer_from_df(
        cls,
        df: pd.DataFrame,
        name: str,
        version: str = "1.0.0",
    ) -> "SchemaDefinition":
        """Infer schema from an observed DataFrame (use on first load)."""
        cols = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            nullable = bool(df[col].isnull().any())
            col_schema = ColumnSchema(name=col, dtype=dtype, nullable=nullable)
            if pd.api.types.is_numeric_dtype(df[col]):
                col_schema.min_val = float(df[col].min())
                col_schema.max_val = float(df[col].max())
            cols.append(col_schema)
        return cls(name=name, version=version, columns=cols)


@dataclass
class DriftEvent:
    column: str
    drift_type: str    # missing_column | type_change | new_nulls | range_violation | new_category
    expected: Any
    observed: Any
    severity: str      # warning | error


class SchemaDriftDetector:
    """
    Compare an incoming DataFrame against a registered SchemaDefinition.

    Usage
    -----
    >>> schema = SchemaDefinition.load("conf/schemas/transactions_v1.json")
    >>> detector = SchemaDriftDetector(schema)
    >>> events = detector.detect(new_df)
    >>> detector.assert_no_errors(events)  # raises if critical drift
    """

    def __init__(self, schema: SchemaDefinition) -> None:
        self.schema = schema

    def detect(self, df: pd.DataFrame) -> list[DriftEvent]:
        events: list[DriftEvent] = []
        schema_cols = {c.name: c for c in self.schema.columns}

        # Check for missing columns
        for col_name, col_schema in schema_cols.items():
            if col_name not in df.columns:
                events.append(DriftEvent(
                    column=col_name,
                    drift_type="missing_column",
                    expected=col_schema.dtype,
                    observed=None,
                    severity="error",
                ))
                continue

            actual = df[col_name]

            # Type drift
            actual_dtype = str(actual.dtype)
            if not _dtypes_compatible(col_schema.dtype, actual_dtype):
                events.append(DriftEvent(
                    column=col_name,
                    drift_type="type_change",
                    expected=col_schema.dtype,
                    observed=actual_dtype,
                    severity="error",
                ))

            # Null drift
            has_nulls = actual.isnull().any()
            if has_nulls and not col_schema.nullable:
                null_pct = round(float(actual.isnull().mean() * 100), 2)
                events.append(DriftEvent(
                    column=col_name,
                    drift_type="new_nulls",
                    expected="not nullable",
                    observed=f"{null_pct}% null",
                    severity="warning",
                ))

            # Range drift (numeric)
            if col_schema.min_val is not None and pd.api.types.is_numeric_dtype(actual):
                obs_min = float(actual.min())
                obs_max = float(actual.max())
                if obs_min < col_schema.min_val * 0.5 or obs_max > col_schema.max_val * 2:
                    events.append(DriftEvent(
                        column=col_name,
                        drift_type="range_violation",
                        expected=f"[{col_schema.min_val}, {col_schema.max_val}]",
                        observed=f"[{obs_min}, {obs_max}]",
                        severity="warning",
                    ))

        # New columns (informational)
        new_cols = [c for c in df.columns if c not in schema_cols]
        for col in new_cols:
            events.append(DriftEvent(
                column=col,
                drift_type="unexpected_column",
                expected=None,
                observed=str(df[col].dtype),
                severity="warning",
            ))

        self._log_events(events)
        return events

    def assert_no_errors(self, events: list[DriftEvent]) -> None:
        errors = [e for e in events if e.severity == "error"]
        if errors:
            msgs = [f"{e.column}: {e.drift_type} (expected={e.expected}, got={e.observed})" for e in errors]
            raise RuntimeError("Schema drift errors detected:\n" + "\n".join(msgs))

    def _log_events(self, events: list[DriftEvent]) -> None:
        if not events:
            log.info("Schema check passed", schema=self.schema.name, version=self.schema.version)
            return
        for e in events:
            if e.severity == "error":
                log.error("Schema drift", **asdict(e))
            else:
                log.warning("Schema drift", **asdict(e))


def _dtypes_compatible(expected: str, observed: str) -> bool:
    """Loose dtype compatibility — int32/int64 are compatible, float32/float64 are compatible."""
    def normalise(d: str) -> str:
        if d.startswith("int"):
            return "int"
        if d.startswith("float"):
            return "float"
        if d.startswith("datetime"):
            return "datetime"
        return d.lower()
    return normalise(expected) == normalise(observed)
