"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    roc_auc,
    gini,
    ks_statistic,
    pr_auc,
    capture_rate_at_decile,
    calibration_error,
    evaluate_all,
)


def test_roc_auc_perfect(binary_preds):
    y_true, _ = binary_preds
    # Perfect predictor
    perfect_score = y_true.astype(float)
    assert roc_auc(y_true, perfect_score) == 1.0


def test_roc_auc_random(binary_preds):
    y_true, _ = binary_preds
    np.random.seed(99)
    random_score = np.random.uniform(0, 1, size=len(y_true))
    auc = roc_auc(y_true, random_score)
    assert 0.4 < auc < 0.6, f"Random classifier AUC should be ~0.5, got {auc}"


def test_gini_range(binary_preds):
    y_true, y_score = binary_preds
    g = gini(y_true, y_score)
    assert -1.0 <= g <= 1.0


def test_ks_positive(binary_preds):
    y_true, y_score = binary_preds
    ks = ks_statistic(y_true, y_score)
    assert ks > 0, "KS should be positive for a model better than random"


def test_capture_rate_top_10(binary_preds):
    y_true, y_score = binary_preds
    rate = capture_rate_at_decile(y_true, y_score, top_pct=0.10)
    # A decent model should capture >20% of positives in top 10%
    assert rate > 0.2, f"Top-10% capture rate too low: {rate}"


def test_evaluate_all_returns_all_keys(binary_preds):
    y_true, y_score = binary_preds
    metrics = evaluate_all(y_true, y_score)
    expected_keys = {"roc_auc", "pr_auc", "gini", "ks", "f1_at_05", "capture_top10pct", "ece"}
    for k in expected_keys:
        assert k in metrics, f"Missing metric: {k}"


def test_calibration_error_range(binary_preds):
    y_true, y_score = binary_preds
    ece = calibration_error(y_true, y_score)
    assert 0.0 <= ece <= 1.0
