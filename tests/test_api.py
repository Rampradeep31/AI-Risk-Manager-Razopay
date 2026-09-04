"""API contract tests for the FastAPI scoring service, using TestClient
(no network/server process needed)."""
import pytest
from fastapi.testclient import TestClient

from scoring_api import app

client = TestClient(app)

VALID_HIGH_RISK_ORDER = {
    "account_age_days": 0, "prior_orders_count": 0, "prior_return_rate": 0.0,
    "order_value": 4500.0, "item_category": "footwear", "item_count": 1,
    "discount_pct": 55.0, "payment_method": "cod", "delivery_promise_days": 3,
    "actual_delivery_days": 7, "device_type": "mobile", "time_of_day": "evening",
    "is_first_order": 1,
}

VALID_LOW_RISK_ORDER = {
    "account_age_days": 900, "prior_orders_count": 40, "prior_return_rate": 0.02,
    "order_value": 4500.0, "item_category": "electronics", "item_count": 1,
    "discount_pct": 5.0, "payment_method": "card", "delivery_promise_days": 3,
    "actual_delivery_days": 3, "device_type": "desktop", "time_of_day": "afternoon",
    "is_first_order": 0,
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_score_high_risk_order_is_flagged():
    resp = client.post("/score", json=VALID_HIGH_RISK_ORDER)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["flagged_for_review"] is True
    assert len(body["top_contributors"]) == 3
    assert isinstance(body["explanation"], str) and len(body["explanation"]) > 0


def test_score_low_risk_order_is_not_flagged():
    resp = client.post("/score", json=VALID_LOW_RISK_ORDER)
    assert resp.status_code == 200
    body = resp.json()
    assert body["flagged_for_review"] is False
    assert body["risk_score"] < 0.5


def test_score_rejects_missing_required_field():
    incomplete = dict(VALID_HIGH_RISK_ORDER)
    del incomplete["order_value"]
    resp = client.post("/score", json=incomplete)
    assert resp.status_code == 422


def test_score_rejects_invalid_category():
    bad = dict(VALID_HIGH_RISK_ORDER)
    bad["item_category"] = "not_a_real_category"
    resp = client.post("/score", json=bad)
    assert resp.status_code == 422


def test_score_rejects_out_of_range_discount():
    bad = dict(VALID_HIGH_RISK_ORDER)
    bad["discount_pct"] = 150.0  # > 100, should fail Field(le=100) validation
    resp = client.post("/score", json=bad)
    assert resp.status_code == 422


def test_score_accepts_null_actual_delivery_days_for_in_transit_orders():
    in_transit = dict(VALID_HIGH_RISK_ORDER)
    in_transit["actual_delivery_days"] = None
    resp = client.post("/score", json=in_transit)
    assert resp.status_code == 200
