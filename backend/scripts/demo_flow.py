"""End-to-end demo against a running server (python scripts/demo_flow.py).

Story: top alert -> explanation -> 'Was this you?' email -> Not me -> freeze + case with deadlines ->
blocked follow-up payment -> another customer confirms 'It was me' with OTP -> trusted pattern lowers friction ->
bank ops works the case -> customer unfreezes with OTP -> model health.
"""
from __future__ import annotations

import os
import re
import sys
import textwrap
from pathlib import Path

import httpx

BASE = os.getenv("API_URL", "http://localhost:8000")
KEY = os.getenv("OPS_API_KEY") or (Path(__file__).resolve().parents[1] / "data" / "ops_api_key.txt").read_text().strip()
OPS = {"X-Ops-Key": KEY}
c = httpx.Client(base_url=BASE, timeout=60)


def h(title):
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


def show(label, value):
    print(textwrap.fill(f"{label}: {value}", 110, subsequent_indent="    "))


def ok(r):
    if r.status_code >= 400:
        sys.exit(f"{r.request.method} {r.request.url} -> {r.status_code} {r.text}")
    return r.json()


def token_from_inbox():
    mail = ok(c.get("/api/demo/inbox", params={"channel": "email", "limit": 5}))["items"]
    body = next(m["body"] for m in mail if (m["subject"] or "").startswith("Was this you?"))
    return re.search(r"token=([A-Za-z0-9_\-]+)", body).group(1), body


def otp_from_inbox(customer_id):
    sms = ok(c.get("/api/demo/inbox", params={"channel": "sms", "customer_id": customer_id, "limit": 1}))["items"][0]
    return re.search(r": (\d{6})\.", sms["body"]).group(1)


def pick(band=None, skip=()):
    page = 1
    while True:
        items = ok(c.get("/api/alerts", params={"status": "NEW", "size": 200, "page": page}))["items"]
        for it in items:
            if it["account_status"] == "active" and it["source"] == "dataset" and it["customer_id"] not in skip and (not band or it["band"] == band):
                return it
        page += 1


h("1. System")
o = ok(c.get("/api/overview"))
show("Transactions", f"{o['transactions']:,} ({o['period'][0]} -> {o['period'][1]})")
show("Alerts", f"{o['alerts']:,} ({o['alerts_per_1000']} per 1,000)")
show("Adaptive-friction bands", o["bands"])
show("Disclaimer", o["disclaimer"])

h("2. Review queue: highest expected loss first")
a = pick(band="BLOCK")
show("Alert", f"{a['transaction_id']} | customer {a['customer_id']} | amount {a['transaction_amount']:,.2f} | "
              f"score {a['score']:.3f} | expected loss {a['expected_loss']:,.2f} | band {a['band']}")
d = ok(c.get(f"/api/alerts/{a['transaction_id']}"))
show("Analyst summary", d["explanation"]["summary"])
show("Numbers verified from data", d["explanation"]["numbers_verified"])
show("Reason families (SHAP, log-odds)", d["explanation"]["families"])
show("Anomaly percentile (not a probability)", d["anomaly_percentile"])
show("Counterfactual", [(x["feature"], x["from"], "->", x["to"]) for x in d["counterfactual"]["changes"]]
     + [f"score {d['counterfactual']['original_score']:.3f} -> {d['counterfactual']['final_score']:.3f}"])
for w in d["what_changed"]:
    show(f"What changed? ({w['family']})", f"{w['original_score']:.3f} -> {w['hypothetical_score']:.3f}  [{w['note']}]")

h("2b. NVIDIA Nemotron on Nebius Token Factory (Agentic Reasoning & Verifier)")
nem_cust = ok(c.post(f"/api/alerts/{a['transaction_id']}/nemotron/customer-message", headers=OPS))
show("Customer alert rewrite", f"source={nem_cust['source']} | verified={nem_cust['verified']} | leakage_passed={nem_cust['leakage_passed']}")
show("Message text", nem_cust["customer_message"])

nem_memo = ok(c.post(f"/api/alerts/{a['transaction_id']}/nemotron/case-memo", headers=OPS))
show("Forensic case memo", f"source={nem_memo['source']} | verified={nem_memo['verified']} | model={nem_memo['model']}")
print(textwrap.indent(nem_memo["case_memo"][:300] + "...", "    | "))

h("3. Bank sends 'Was this you?' (customer-safe wording, single-use link)")
ok(c.post(f"/api/ops/alerts/{a['transaction_id']}/notify", headers=OPS))
token, body = token_from_inbox()
print(textwrap.indent(body.strip(), "    | "))
info = ok(c.post("/api/respond/info", json={"token": token}))
show("Opening the link changes nothing", f"link_usable={info['link_usable']} options={info['options']}")

h("4. Customer clicks 'Not me' -> freeze (simulated) + case with RBI-style deadlines")
r = ok(c.post("/api/respond/not-me", json={"token": token}))
show("Case", f"#{r['case']['case_id']} {r['case']['status']} | account {r['account']['status']}")
show("Deadlines", {k: r["sla"][k] for k in ("zero_liability", "shadow_credit_due", "resolution_due")})
for act in r["actions"]:
    show(f"  {act['actor']:<8} {act['action']}", act["note"])
again = ok(c.post("/api/respond/not-me", json={"token": token}))
show("Double click", f"idempotent_replay={again['idempotent_replay']} same case={again['case']['case_id'] == r['case']['case_id']}")

h("5. Attacker tries another payment from the frozen account")
p = ok(c.get("/api/simulate/presets", params={"customer_id": a["customer_id"]}))["presets"]["typical"]
s = ok(c.post("/api/simulate", json=p))
show("Decision", f"{s['decision']['band']} - {s['decision']['notes'][-1]}")

h("6. A different customer: 'Yes, it was me' needs an OTP -> trusted pattern")
b = pick(band="HOLD", skip={a["customer_id"]})
before = ok(c.get(f"/api/transactions/{b['transaction_id']}"))["decision"]
show("Before", f"{b['transaction_id']} band {before['band']}")
ok(c.post(f"/api/ops/alerts/{b['transaction_id']}/notify", headers=OPS))
tok_b, _ = token_from_inbox()
show("OTP sent", ok(c.post("/api/respond/was-me/start", json={"token": tok_b})))
conf = ok(c.post("/api/respond/was-me/confirm", json={"token": tok_b, "otp": otp_from_inbox(b["customer_id"])}))
show("Trusted pattern", conf["trusted_pattern"])
after = ok(c.get(f"/api/transactions/{b['transaction_id']}"))["decision"]
show("After", f"band {after['model_band']} -> {after['band']} | {after['notes'][-1]}")

h("7. Bank operations work the case")
cid = r["case"]["case_id"]
for step, note in (("INVESTIGATING", "Reviewing device and merchant records"), ("SHADOW_CREDITED", "Provisional credit issued")):
    v = ok(c.post(f"/api/ops/cases/{cid}/transition", json={"to": step, "note": note}, headers=OPS))
    show(step, f"allowed next: {v['allowed_transitions']}")
bad = c.post(f"/api/ops/cases/{cid}/transition", json={"to": "INVESTIGATING"}, headers=OPS)
show("Invalid transition", f"{bad.status_code} {bad.json()['error']['message']}")
show("Without X-Ops-Key", c.get("/api/ops/cases").status_code)

h("8. Customer unfreezes with an OTP from the registered phone")
ok(c.post(f"/api/customers/{a['customer_id']}/unfreeze/start"))
u = ok(c.post(f"/api/customers/{a['customer_id']}/unfreeze/confirm", json={"otp": otp_from_inbox(a["customer_id"])}))
show("Account", u["account"]["status"])

h("9. Model health and measured results")
m = ok(c.get("/api/monitoring"))
show("Drift status (PSI)", f"{m['status']} (reference month {m['reference_month']})")
ev = ok(c.get("/api/evaluation"))["evaluation"]
for k in ("catboost", "isolation_forest", "amount_threshold_baseline"):
    e = ev[k]
    show(k, f"AP {e['avg_precision']} | ROC-AUC {e['roc_auc']} | precision {e['precision']} | recall {e['recall']} | alerts/1000 {e['alerts_per_1000']}")
print("\nDone. Explore everything at", BASE + "/docs")
