"""FastAPI application: Explainable Suspicious Transaction Detector (synthetic data, simulated outcomes)."""
from __future__ import annotations

import csv
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import clock, decision, monitoring, nemotron, reasons
from .config import settings
from .db import Database, rows
from .model_service import ModelService
from .schemas import NoteBody, OtpBody, SimulateRequest, TokenBody, TokenOtpBody, TransitionBody
from .security import limiter, request_meta, require_ops_key
from .workflow import AppError, Engine, _public_txn, _py
from .easata_mail import router as easata_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("stdetector")

DISCLAIMER = "Synthetic data. Scores mean 'flagged for review', not proof of fraud. All outcomes are simulated."


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = Database(settings.db_path)
    db.init_schema()
    model = ModelService(settings.bundle_dir)
    if not db.transactions_loaded():
        log.info("first start: loading demo transactions from the bundle (takes a few seconds)")
        db.load_transactions(settings.bundle_dir / "demo_transactions.csv.gz", settings.demo_max_rows)
    engine = Engine(db, model)
    with db.read() as conn:
        app.state.options = {c: [r[0] for r in conn.execute(f"SELECT DISTINCT {c} FROM transactions ORDER BY {c}")]
                             for c in ("country", "city", "merchant_category", "payment_method", "device_type")}
    app.state.db, app.state.model, app.state.engine = db, model, engine
    key = settings.resolve_ops_key()
    log.info("ready | threshold=%.4f | expected-loss-high=%.2f | ops key: %s", model.threshold, engine.el_high,
             "from OPS_API_KEY" if key and not (settings.db_path.parent / 'ops_api_key.txt').exists() else
             f"saved in {settings.db_path.parent / 'ops_api_key.txt'}")
    yield


app = FastAPI(
    title="EASATA – Every alert has a story API",
    version="1.0.0",
    description=DISCLAIMER + "\n\nBank-operations endpoints (/api/ops/*) need the `X-Ops-Key` header. "
                             "Customer endpoints (/api/respond/*, unfreeze) are protected by single-use link tokens and OTPs.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-Ops-Key", "Idempotency-Key", "X-Request-ID"])


app.include_router(easata_router)


@app.middleware("http")
async def headers_and_logging(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"          # tokens in /respond URLs never leak via Referer
    if request.url.path.startswith("/api/respond") or "/unfreeze" in request.url.path:
        response.headers["Cache-Control"] = "no-store"
    log.info("%s %s %s %.0fms rid=%s", request.method, request.url.path, response.status_code,
             (time.perf_counter() - t0) * 1000, rid)
    return response


def _err(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, **extra}})


@app.exception_handler(AppError)
async def app_error(_: Request, e: AppError):
    return _err(e.status, e.code, e.message, **e.extra)


@app.exception_handler(HTTPException)
async def http_error(_: Request, e: HTTPException):
    if isinstance(e.detail, dict):
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    return _err(e.status_code, "http_error", str(e.detail))


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, e: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in err["loc"][1:]), "message": err["msg"]} for err in e.errors()]
    return _err(422, "validation_error", "Request validation failed", details=details)


def eng(request: Request) -> Engine:
    return request.app.state.engine


# ============================================================ system
@app.get("/api/health", tags=["system"])
def health(request: Request):
    m: ModelService = request.app.state.model
    with request.app.state.db.read() as conn:
        n = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    return {"status": "ok", "transactions": n, "model_loaded": True, "warnings": m.warnings, "time": clock.iso()}


@app.get("/api/meta", tags=["system"], summary="Model info, form options and settings for the frontend")
def meta(request: Request):
    m: ModelService = request.app.state.model
    e: Engine = request.app.state.engine
    return {
        "disclaimer": DISCLAIMER, "dataset": m.cfg.get("dataset"), "threshold": m.threshold,
        "threshold_mode": m.cfg.get("threshold_mode"), "expected_loss_high": e.el_high,
        "bands": decision.BANDS, "band_labels": decision.LABELS, "families": sorted(set(m.family.values())),
        "features": m.features, "versions": m.cfg.get("versions"), "split": m.cfg.get("split"),
        "options": request.app.state.options, "amount_quantiles": e.amount_quantiles,
        "token_ttl_min": settings.token_ttl_min, "otp_ttl_min": settings.otp_ttl_min, "demo_mode": settings.demo_mode,
        "warnings": m.warnings,
    }


# ============================================================ dashboard reads
@app.get("/api/overview", tags=["dashboard"])
def overview(request: Request):
    e: Engine = eng(request)
    with request.app.state.db.read() as conn:
        df = pd.read_sql("SELECT score, transaction_amount, alert, is_fraud, ts FROM transactions WHERE source='dataset'", conn)
        wf = {
            "alerts_by_status": {r["status"]: r["n"] for r in rows(conn.execute("SELECT status, COUNT(*) n FROM alerts GROUP BY status"))},
            "frozen_accounts": conn.execute("SELECT COUNT(*) FROM accounts WHERE status='frozen'").fetchone()[0],
            "cases_by_status": {r["status"]: r["n"] for r in rows(conn.execute("SELECT status, COUNT(*) n FROM cases GROUP BY status"))},
            "feedback": {r["response"]: r["n"] for r in rows(conn.execute("SELECT response, COUNT(*) n FROM feedback GROUP BY response"))},
            "active_trusted_patterns": conn.execute("SELECT COUNT(*) FROM trusted_patterns WHERE revoked_at IS NULL AND expires_at>?",
                                                    (clock.iso(),)).fetchone()[0],
            "simulated_transactions": conn.execute("SELECT COUNT(*) FROM transactions WHERE source='simulated'").fetchone()[0],
        }
    lv = decision.levels_vectorized(df.score, df.transaction_amount, e.model.threshold, e.el_high)
    bands = {b: int((lv == i).sum()) for i, b in enumerate(decision.BANDS)}
    return {
        "disclaimer": DISCLAIMER,
        "transactions": len(df), "period": [df.ts.min(), df.ts.max()],
        "alerts": int(df.alert.sum()), "alerts_per_1000": round(1000 * float(df.alert.mean()), 2),
        "bands": bands, "expected_loss_flagged_total": round(float((df.score * df.transaction_amount)[df.alert == 1].sum()), 2),
        "synthetic_label_rate": round(float(df.is_fraud.mean()), 5),
        "workflow": wf,
        "model": {"test_avg_precision": e.model.evaluation["catboost"]["avg_precision"],
                  "test_precision": e.model.evaluation["catboost"]["precision"],
                  "test_recall": e.model.evaluation["catboost"]["recall"]},
    }


TXN_LIST_COLS = ("transaction_id, customer_id, ts, transaction_amount, merchant_category, payment_method, device_type, "
                 "country, city, is_international, failed_attempts, pin_changed_recently, score, alert, anomaly_pct, source, status")


@app.get("/api/transactions", tags=["dashboard"])
def list_transactions(request: Request, customer_id: str | None = None, alert: bool | None = None,
                      min_score: float | None = Query(None, ge=0, le=1), merchant_category: str | None = None,
                      source: Literal["dataset", "simulated"] | None = None, date_from: str | None = None, date_to: str | None = None,
                      sort: Literal["ts", "-ts", "score", "-score", "amount", "-amount"] = "-ts",
                      page: int = Query(1, ge=1), size: int = Query(50, ge=1, le=500)):
    where, args = [], []
    for cond, val in (("customer_id=?", customer_id), ("merchant_category=?", merchant_category), ("source=?", source),
                      ("ts>=?", date_from), ("ts<=?", date_to), ("score>=?", min_score)):
        if val is not None:
            where.append(cond)
            args.append(val)
    if alert is not None:
        where.append("alert=?")
        args.append(int(alert))
    w = ("WHERE " + " AND ".join(where)) if where else ""
    order = {"ts": "ts ASC", "-ts": "ts DESC", "score": "score ASC", "-score": "score DESC",
             "amount": "transaction_amount ASC", "-amount": "transaction_amount DESC"}[sort]
    with request.app.state.db.read() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM transactions {w}", args).fetchone()[0]
        items = rows(conn.execute(f"SELECT {TXN_LIST_COLS} FROM transactions {w} ORDER BY {order}, rowid LIMIT ? OFFSET ?",
                                  (*args, size, (page - 1) * size)))
    e = eng(request)
    for it in items:
        it["band"] = decision.BANDS[int(decision.levels_vectorized([it["score"]], [it["transaction_amount"]], e.model.threshold, e.el_high)[0])]
    return {"total": total, "page": page, "size": size, "items": items}


@app.get("/api/transactions/{transaction_id}", tags=["dashboard"])
def get_transaction(transaction_id: str, request: Request):
    e = eng(request)
    with request.app.state.db.read() as conn:
        row = e.txn(conn, transaction_id)
        return {"transaction": _public_txn(row), "score": e.model.score(row), "decision": e.decide_for(conn, row, e.model.score(row))}


@app.get("/api/alerts", tags=["alerts"], summary="Review queue (default: sorted by expected loss = probability x amount)")
def list_alerts(request: Request, status: Literal["ALL", "NEW", "NOTIFIED", "CONFIRMED_NOT_ME", "CONFIRMED_LEGIT"] = "ALL",
                sort: Literal["expected_loss", "score", "ts", "amount"] = "expected_loss",
                page: int = Query(1, ge=1), size: int = Query(50, ge=1, le=500)):
    e = eng(request)
    where, args = ["t.score>=?", "t.status='completed'"], [e.model.threshold]
    if status != "ALL":
        where.append("COALESCE(a.status,'NEW')=?")
        args.append(status)
    w = " AND ".join(where)
    order = {"expected_loss": "expected_loss DESC", "score": "t.score DESC", "ts": "t.ts DESC", "amount": "t.transaction_amount DESC"}[sort]
    base = ("FROM transactions t LEFT JOIN alerts a ON a.transaction_id=t.transaction_id "
            "LEFT JOIN accounts acc ON acc.customer_id=t.customer_id WHERE " + w)
    with request.app.state.db.read() as conn:
        total = conn.execute(f"SELECT COUNT(*) {base}", args).fetchone()[0]
        items = rows(conn.execute(
            "SELECT t.transaction_id, t.customer_id, t.ts, t.transaction_amount, t.merchant_category, t.payment_method, t.device_type, "
            "t.country, t.city, t.score, t.anomaly_pct, t.source, t.score*t.transaction_amount AS expected_loss, "
            f"COALESCE(a.status,'NEW') AS alert_status, a.alert_id, COALESCE(acc.status,'active') AS account_status {base} "
            f"ORDER BY {order}, t.rowid LIMIT ? OFFSET ?", (*args, size, (page - 1) * size)))
    for it in items:
        it["band"] = decision.decide(it["score"], it["transaction_amount"], e.model.threshold, e.el_high,
                                     frozen=it["account_status"] == "frozen")["band"]
    return {"total": total, "page": page, "size": size, "sort": sort, "items": items}


@app.get("/api/alerts/{transaction_id}", tags=["alerts"],
         summary="Full analysis: score, decision band, verified reasons, SHAP, what-changed, counterfactual, history")
def alert_detail(transaction_id: str, request: Request):
    return eng(request).analyse(transaction_id)


# ============================================================ NVIDIA Nemotron on Nebius Token Factory
@app.post("/api/alerts/{transaction_id}/nemotron/customer-message", tags=["nemotron"],
          dependencies=[Depends(require_ops_key)],
          summary="Customer-safe alert rewrite generated by NVIDIA Nemotron on Nebius with leakage and verifier checks")
def nemotron_customer_message(transaction_id: str, request: Request):
    limiter.check(request, "nemotron", 60)
    e: Engine = eng(request)
    with request.app.state.db.read() as conn:
        row = e.txn(conn, transaction_id)
        expl = e.model.explain(row)
    base = e.model.baseline
    raw_reasons = expl.get("reasons", [])
    cust_msg_obj = reasons.customer_message(row, raw_reasons, base)
    res = nemotron.generate_customer_message(
        row=row,
        reasons_list=raw_reasons,
        base=base,
        fallback_template=cust_msg_obj.get("text", expl.get("customer_message", "")),
        evidence=cust_msg_obj.get("evidence", []),
    )
    with request.app.state.db.write() as conn:
        nemotron.record_generation(
            conn,
            transaction_id=transaction_id,
            generation_type="customer_message",
            source=res["source"],
            model=res["model"],
            output_text=res["customer_message"],
            latency_ms=res["latency_ms"],
            tokens=res["tokens"].get("total_tokens", 0) if res.get("tokens") else 0,
            verified=res["verified"],
            leakage_passed=res["leakage_passed"],
            fallback_reason=res.get("fallback_reason"),
        )
    return {"transaction_id": transaction_id, **res}


@app.post("/api/alerts/{transaction_id}/nemotron/case-memo", tags=["nemotron"],
          dependencies=[Depends(require_ops_key)],
          summary="Forensic case investigation memo generated by NVIDIA Nemotron on Nebius (RBI Zero-Liability SLA aligned)")
def nemotron_case_memo(transaction_id: str, request: Request):
    limiter.check(request, "nemotron", 60)
    e: Engine = eng(request)
    analysis = e.analyse(transaction_id)
    row = analysis["transaction"]
    expl = analysis["explanation"]
    with request.app.state.db.read() as conn:
        actions = rows(conn.execute(
            "SELECT ca.action, ca.actor, ca.note, ca.created_at FROM case_actions ca "
            "JOIN cases c USING(case_id) WHERE c.transaction_id=? ORDER BY ca.id", (transaction_id,)
        ))
    res = nemotron.generate_case_memo(
        row=row,
        score=analysis["score"],
        threshold=expl["threshold"],
        families=expl.get("families", {}),
        counterfactual=analysis.get("counterfactual", {}),
        case=analysis.get("case"),
        actions=actions,
        base=e.model.baseline,
    )
    with request.app.state.db.write() as conn:
        nemotron.record_generation(
            conn,
            transaction_id=transaction_id,
            generation_type="case_memo",
            source=res["source"],
            model=res["model"],
            output_text=res["case_memo"],
            latency_ms=res["latency_ms"],
            tokens=res["tokens"].get("total_tokens", 0) if res.get("tokens") else 0,
            verified=res["verified"],
            leakage_passed=True,
            fallback_reason=res.get("fallback_reason"),
        )
    return {"transaction_id": transaction_id, **res}


@app.get("/api/nemotron/status", tags=["nemotron"], dependencies=[Depends(require_ops_key)],
         summary="Status of Nebius Token Factory connection, model availability, and recent generations")
def nemotron_status(request: Request):
    key_configured = bool(settings.nebius_api_key)
    resolved_model, is_available = nemotron.resolve_model()
    nem_models = nemotron.get_nemotron_models() if key_configured else []
    with request.app.state.db.read() as conn:
        recent = rows(conn.execute(
            "SELECT transaction_id, generation_type, source, model, latency_ms, tokens, "
            "verified, leakage_passed, fallback_reason, created_at "
            "FROM nemotron_generations ORDER BY id DESC LIMIT 5"
        ))
    return {
        "key_configured": key_configured,
        "base_url": settings.nebius_base_url,
        "configured_model": settings.nebius_model,
        "resolved_model": resolved_model,
        "model_available": is_available,
        "nemotron_models_in_catalog": nem_models,
        "recent_generations": recent,
    }


@app.get("/api/customers/{customer_id}", tags=["customers"])
def customer(customer_id: str, request: Request, limit: int = Query(100, ge=1, le=1000)):
    e = eng(request)
    with request.app.state.db.read() as conn:
        profile = e.customer_profile(conn, customer_id)
        if profile is None:
            raise AppError(404, "customer_not_found", f"No transactions for {customer_id}")
        return {
            "customer_id": customer_id, "profile": profile, "account": e.account(conn, customer_id),
            "trusted_patterns": e.trusted_for(conn, customer_id),
            "history": e.customer_history(conn, customer_id, limit=limit),
            "cases": rows(conn.execute("SELECT case_id, status, transaction_id, reported_at FROM cases WHERE customer_id=? ORDER BY case_id DESC", (customer_id,))),
            "feedback": rows(conn.execute("SELECT * FROM feedback WHERE customer_id=? ORDER BY created_at DESC", (customer_id,))),
        }


@app.get("/api/replay", tags=["dashboard"], summary="Time-ordered stream of dataset transactions with a cursor")
def replay(request: Request, cursor: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500)):
    e = eng(request)
    with request.app.state.db.read() as conn:
        items = rows(conn.execute(f"SELECT rowid AS cursor, {TXN_LIST_COLS} FROM transactions WHERE source='dataset' AND rowid>? "
                                  "ORDER BY rowid LIMIT ?", (cursor, limit)))
    if items:
        lv = decision.levels_vectorized([i["score"] for i in items], [i["transaction_amount"] for i in items], e.model.threshold, e.el_high)
        for it, l in zip(items, lv):
            it["band"] = decision.BANDS[int(l)]
    return {"items": items, "next_cursor": items[-1]["cursor"] if items else cursor, "done": len(items) < limit}


@app.get("/api/evaluation", tags=["model"], summary="Measured test results from the training notebook (not hardcoded)")
def evaluation(request: Request):
    m: ModelService = request.app.state.model
    return {"evaluation": m.evaluation, "threshold": m.threshold, "threshold_mode": m.cfg.get("threshold_mode"),
            "best_params": m.cfg.get("best_params"), "split": m.cfg.get("split"), "dataset": m.cfg.get("dataset"),
            "notes": ["Time-based split; threshold chosen on validation; metrics on the untouched test period.",
                      "Anomaly (Isolation Forest) is a percentile, not a probability.", DISCLAIMER]}


@app.get("/api/monitoring", tags=["model"], summary="Monthly alert rate / score and feature drift (PSI)")
def monitoring_view(request: Request):
    m: ModelService = request.app.state.model
    top = [f for f in list(m.evaluation.get("global_importance", {}).keys()) if f in m.features][:8]
    return monitoring.compute(request.app.state.db, m.features, m.cats, m.threshold, top)


# ============================================================ simulation
@app.get("/api/simulate/presets", tags=["simulate"], summary="Editable example inputs built from this customer's data")
def presets(request: Request, customer_id: str):
    e = eng(request)
    with request.app.state.db.read() as conn:
        prof = e.customer_profile(conn, customer_id)
        if prof is None:
            raise AppError(404, "customer_not_found", f"No transactions for {customer_id}")
        mode = lambda c: conn.execute(f"SELECT {c} FROM transactions WHERE customer_id=? GROUP BY {c} ORDER BY COUNT(*) DESC LIMIT 1",
                                      (customer_id,)).fetchone()[0]
        typical = {c: mode(c) for c in ("merchant_category", "payment_method", "device_type", "country", "city")}
    q, base = e.amount_quantiles, e.model.baseline
    last = pd.Timestamp(prof["ts"])
    night = (last + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=2, minutes=14)
    common = {"customer_id": customer_id}
    return {"customer_id": customer_id, "last_transaction_ts": prof["ts"], "presets": {
        "typical": {**common, **typical, "transaction_amount": round(q["p50"], 2), "distance_from_home_km": round(float(base["distance_from_home_km"]), 1),
                    "is_international": 0, "failed_attempts": 0, "pin_changed_recently": 0,
                    "timestamp": (last + pd.Timedelta(hours=26)).replace(hour=14, minute=5).strftime("%Y-%m-%d %H:%M:%S")},
        "account_takeover_like": {**common, **typical, "merchant_category": "Crypto Exchange", "device_type": "Mobile",
                                  "transaction_amount": round(q["p95"] * 3, 2), "distance_from_home_km": 1500.0,
                                  "is_international": 1, "failed_attempts": 3, "pin_changed_recently": 1,
                                  "timestamp": night.strftime("%Y-%m-%d %H:%M:%S")},
        "travel_legit_like": {**common, **typical, "merchant_category": "Travel", "transaction_amount": round(q["p90"], 2),
                              "distance_from_home_km": 2000.0, "is_international": 1, "failed_attempts": 0, "pin_changed_recently": 0,
                              "timestamp": (last + pd.Timedelta(hours=30)).strftime("%Y-%m-%d %H:%M:%S")},
    }, "note": "Scenario inputs you can edit. Amounts come from this dataset's percentiles; they are not real customer data."}


@app.post("/api/simulate", tags=["simulate"],
          summary="Score a new transaction. commit=false: pending (nothing stored). commit=true: stored + workflow runs.")
def simulate(body: SimulateRequest, request: Request, commit: bool = False,
             idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    opts = request.app.state.options
    bad = {c: opts[c] for c in ("merchant_category", "payment_method", "device_type", "country", "city") if getattr(body, c) not in opts[c]}
    if bad:
        raise AppError(422, "unknown_category", "Unknown category value(s); use one of the allowed values.",
                       {"allowed": bad})
    limiter.check(request, "simulate", 120)
    e = eng(request)
    payload = body.model_dump()
    if commit:
        return e.idempotent(idempotency_key, "simulate", payload, lambda: e.simulate(payload, True))
    return e.simulate(payload, False)


# ============================================================ customer: "Was this you?"
@app.post("/api/respond/info", tags=["customer"], summary="What the /respond page shows. Read-only (safe for link scanners).")
def respond_info(body: TokenBody, request: Request):
    limiter.check(request, "respond")
    return eng(request).respond_info(body.token)


@app.post("/api/respond/not-me", tags=["customer"], summary="Customer reports the transaction: freeze (simulated) + open case")
def respond_not_me(body: TokenBody, request: Request):
    limiter.check(request, "respond")
    return eng(request).not_me(body.token, request_meta(request))


@app.post("/api/respond/was-me/start", tags=["customer"], summary="Send an OTP to the registered phone (simulated SMS)")
def respond_was_me_start(body: TokenBody, request: Request):
    limiter.check(request, "respond")
    return eng(request).was_me_start(body.token)


@app.post("/api/respond/was-me/confirm", tags=["customer"], summary="Confirm with OTP: closes alert and creates a trusted pattern")
def respond_was_me_confirm(body: TokenOtpBody, request: Request):
    limiter.check(request, "respond")
    return eng(request).was_me_confirm(body.token, body.otp)


@app.post("/api/customers/{customer_id}/unfreeze/start", tags=["customer"])
def unfreeze_start(customer_id: str, request: Request):
    limiter.check(request, "unfreeze")
    return eng(request).unfreeze_start(customer_id)


@app.post("/api/customers/{customer_id}/unfreeze/confirm", tags=["customer"])
def unfreeze_confirm(customer_id: str, body: OtpBody, request: Request):
    limiter.check(request, "unfreeze")
    return eng(request).unfreeze_confirm(customer_id, body.otp, request_meta(request))


@app.delete("/api/customers/{customer_id}/trusted-patterns/{pattern_id}", tags=["customer"],
            summary="Revoke a trusted pattern (always allowed: it only adds friction)")
def revoke_trusted(customer_id: str, pattern_id: int, request: Request):
    return eng(request).revoke_trusted(customer_id, pattern_id)


# ============================================================ bank operations (X-Ops-Key)
@app.post("/api/ops/alerts/{transaction_id}/notify", tags=["bank ops"], dependencies=[Depends(require_ops_key)],
          summary="Send the 'Was this you?' email for an alert")
def ops_notify(transaction_id: str, request: Request):
    return eng(request).notify(transaction_id)


@app.get("/api/ops/cases", tags=["bank ops"], dependencies=[Depends(require_ops_key)])
def ops_cases(request: Request, status: str | None = None, page: int = Query(1, ge=1), size: int = Query(50, ge=1, le=200)):
    return eng(request).list_cases(status, page, size)


@app.get("/api/ops/cases/{case_id}", tags=["bank ops"], dependencies=[Depends(require_ops_key)])
def ops_case(case_id: int, request: Request):
    return eng(request).get_case(case_id)


@app.post("/api/ops/cases/{case_id}/transition", tags=["bank ops"], dependencies=[Depends(require_ops_key)])
def ops_transition(case_id: int, body: TransitionBody, request: Request):
    return eng(request).transition(case_id, body.to, body.note)


@app.post("/api/ops/cases/{case_id}/notes", tags=["bank ops"], dependencies=[Depends(require_ops_key)])
def ops_note(case_id: int, body: NoteBody, request: Request):
    return eng(request).add_note(case_id, body.note)


@app.get("/api/ops/audit", tags=["bank ops"], dependencies=[Depends(require_ops_key)], summary="Append-only case action log")
def ops_audit(request: Request, limit: int = Query(200, ge=1, le=2000)):
    with request.app.state.db.read() as conn:
        return {"items": rows(conn.execute("SELECT * FROM case_actions ORDER BY id DESC LIMIT ?", (limit,)))}


@app.get("/api/ops/feedback.csv", tags=["bank ops"], dependencies=[Depends(require_ops_key)],
         summary="Customer-confirmed labels with model features, for retraining in Colab")
def ops_feedback_csv(request: Request):
    m: ModelService = request.app.state.model
    cols = ", ".join(f"t.{f}" for f in m.features)
    with request.app.state.db.read() as conn:
        data = rows(conn.execute(f"SELECT f.transaction_id, f.customer_id, f.response, f.created_at, "
                                 f"CASE f.response WHEN 'not_me' THEN 1 ELSE 0 END AS customer_label, t.ts, {cols} "
                                 "FROM feedback f JOIN transactions t USING(transaction_id) ORDER BY f.created_at"))
    buf = io.StringIO()
    header = ["transaction_id", "customer_id", "response", "created_at", "customer_label", "ts", *m.features]
    w = csv.DictWriter(buf, fieldnames=header)
    w.writeheader()
    for r in data:
        w.writerow(r)
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=feedback_labels.csv"})


# ============================================================ demo helpers
@app.get("/api/demo/inbox", tags=["demo"], summary="Emails and SMS the system sent (demo only; a real system has no such endpoint)")
def demo_inbox(request: Request, channel: Literal["email", "sms"] | None = None, customer_id: str | None = None,
               limit: int = Query(50, ge=1, le=500)):
    if not settings.demo_mode:
        raise AppError(404, "not_found", "Not available")
    where, args = [], []
    if channel:
        where.append("channel=?")
        args.append(channel)
    if customer_id:
        where.append("customer_id=?")
        args.append(customer_id)
    w = ("WHERE " + " AND ".join(where)) if where else ""
    with request.app.state.db.read() as conn:
        return {"items": rows(conn.execute(f"SELECT * FROM outbox {w} ORDER BY id DESC LIMIT ?", (*args, limit)))}


# ============================================================ built frontend (optional)
if settings.frontend_dist.exists():
    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = (settings.frontend_dist / path).resolve()
        if path and f.is_file() and settings.frontend_dist.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(settings.frontend_dist / "index.html")
