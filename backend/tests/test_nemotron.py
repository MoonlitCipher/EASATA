"""Tests for NVIDIA Nemotron on Nebius Token Factory:
1. Missing-key fallback
2. Timeout/failure fallback
3. Leakage-guard rejection
4. Verifier rejection of invented numbers
5. Success path with verified outputs
6. Case memo generation and RBI SLA alignment
7. Ops authorization requirement
"""
from unittest.mock import patch
import pytest

from app.config import settings
from conftest import OPS, fresh_alert


def test_nemotron_endpoints_require_ops_key(client):
    a = fresh_alert(client)
    tid = a["transaction_id"]
    r1 = client.post(f"/api/alerts/{tid}/nemotron/customer-message")
    assert r1.status_code == 401
    r2 = client.post(f"/api/alerts/{tid}/nemotron/case-memo")
    assert r2.status_code == 401


def test_nemotron_missing_key_fallback(client):
    a = fresh_alert(client)
    tid = a["transaction_id"]
    with patch.object(settings, "nebius_api_key", ""):
        r = client.post(f"/api/alerts/{tid}/nemotron/customer-message", headers=OPS)
        assert r.status_code == 200
        data = r.json()
        assert data["source"] == "template"
        assert data["verified"] is True
        assert data["leakage_passed"] is True
        assert data["fallback_reason"] == "missing_api_key"
        assert len(data["customer_message"]) > 10


def test_nemotron_timeout_failure_fallback(client):
    a = fresh_alert(client)
    tid = a["transaction_id"]
    with patch("app.nemotron.call_nebius_llm", return_value=(None, 15000.0, {}, "api_error: TimeoutException")):
        r = client.post(f"/api/alerts/{tid}/nemotron/customer-message", headers=OPS)
        assert r.status_code == 200
        data = r.json()
        assert data["source"] == "template"
        assert "api_error" in data["fallback_reason"]
        assert data["customer_message"]


def test_nemotron_leakage_guard_rejection(client):
    """If Nemotron attempts to disclose model internals (score, threshold, SHAP), leakage guard blocks it."""
    a = fresh_alert(client)
    tid = a["transaction_id"]
    leaked_output = "We flagged your payment because your fraud score is 0.85 above the threshold."
    with patch("app.nemotron.call_nebius_llm", return_value=(leaked_output, 320.0, {"total_tokens": 45}, None)):
        r = client.post(f"/api/alerts/{tid}/nemotron/customer-message", headers=OPS)
        assert r.status_code == 200
        data = r.json()
        assert data["source"] == "template"
        assert data["leakage_passed"] is False
        assert "leakage_guard_triggered" in data["fallback_reason"]


def test_nemotron_verifier_rejects_invented_numbers(client):
    """If Nemotron hallucinates an amount or number not in data, verifier blocks it."""
    a = fresh_alert(client)
    tid = a["transaction_id"]
    invented_output = "A payment of 9999999.00 was attempted from a new device."
    with patch("app.nemotron.call_nebius_llm", return_value=(invented_output, 280.0, {"total_tokens": 30}, None)):
        r = client.post(f"/api/alerts/{tid}/nemotron/customer-message", headers=OPS)
        assert r.status_code == 200
        data = r.json()
        assert data["source"] == "template"
        assert data["verified"] is False
        assert data["fallback_reason"] == "verifier_failed_invented_numbers"


def test_nemotron_success_path_verified(client):
    """Compliant Nemotron output with matching numbers passes both leakage check and verifier."""
    a = fresh_alert(client)
    tid = a["transaction_id"]
    amt = a["transaction_amount"]
    valid_text = (
        f"A {a['payment_method']} payment of {amt:,.2f} to a {a['merchant_category']} merchant "
        f"on {a['ts']} ({a['city']}, {a['country']}) from a {a['device_type']} device."
    )
    tokens = {"prompt_tokens": 120, "completion_tokens": 35, "total_tokens": 155}
    with patch("app.nemotron.call_nebius_llm", return_value=(valid_text, 410.5, tokens, None)):
        r = client.post(f"/api/alerts/{tid}/nemotron/customer-message", headers=OPS)
        assert r.status_code == 200
        data = r.json()
        assert data["source"] == "nemotron"
        assert data["verified"] is True
        assert data["leakage_passed"] is True
        assert data["customer_message"] == valid_text
        assert data["tokens"]["total_tokens"] == 155


def test_nemotron_case_memo_generation_and_storage(client):
    a = fresh_alert(client)
    tid = a["transaction_id"]
    r = client.post(f"/api/alerts/{tid}/nemotron/case-memo", headers=OPS)
    assert r.status_code == 200
    data = r.json()
    assert "case_memo" in data
    assert data["source"] in ("nemotron", "template")
    assert "model" in data

    # Verify generation is persisted in nemotron_generations table
    db = client.app.state.db
    with db.read() as conn:
        row = conn.execute("SELECT * FROM nemotron_generations WHERE transaction_id=? ORDER BY id DESC LIMIT 1",
                           (tid,)).fetchone()
        assert row is not None
        assert row["generation_type"] == "case_memo"
        assert row["model"] == settings.nebius_model


def test_nemotron_status_endpoint(client):
    r = client.get("/api/nemotron/status", headers=OPS)
    assert r.status_code == 200
    data = r.json()
    assert "key_configured" in data
    assert "base_url" in data
    assert "configured_model" in data
    assert "recent_generations" in data
    assert isinstance(data["recent_generations"], list)
