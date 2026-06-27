from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.severity import (
    compute_low_confidence_penalty,
    compute_severity,
    compute_shrinkage_score,
    compute_theft_suspicion_indicator,
    route_severity,
)


def test_shrinkage_score_positive_gap():
    assert compute_shrinkage_score(expected_stock=100, cv_count=70) == 0.3


def test_shrinkage_score_zero_when_no_gap():
    assert compute_shrinkage_score(expected_stock=100, cv_count=120) == 0.0


def test_shrinkage_score_expected_stock_zero_no_division_error():
    assert compute_shrinkage_score(expected_stock=0, cv_count=0) == 0.0
    assert compute_shrinkage_score(expected_stock=0, cv_count=-1) == 1.0


def test_theft_suspicion_indicator():
    assert compute_theft_suspicion_indicator(0.2, 0.4, 0.9) == 1
    assert compute_theft_suspicion_indicator(0.5, 0.4, 0.9) == 0
    assert compute_theft_suspicion_indicator(0.2, 0.4, 0.7) == 0


def test_low_confidence_penalty():
    assert compute_low_confidence_penalty(0.9) == 0.0
    assert round(compute_low_confidence_penalty(0.7), 2) == 0.15


def test_severity_formula_and_routing():
    severity = compute_severity(
        demand_anomaly_score=0.4,
        shrinkage_score=0.5,
        theft_suspicion_indicator=0,
        low_confidence_penalty=0.1,
    )
    assert round(severity, 2) == 0.35
    assert route_severity(severity) == "requires_review"


def test_theft_routes_to_freeze():
    assert route_severity(0.2, theft_suspicion_indicator=1) == "freeze_and_alert"
