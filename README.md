# EASATA

### Every alert has a story.
https://easata.onrender.com/
An explainable transaction-risk demo that turns a suspicious payment into a verified explanation and a customer response workflow, from **"Was this you?"** to a simulated freeze and case.

<p align="center">
  <a href="https://easata.onrender.com"><img src="https://img.shields.io/badge/OPEN_LIVE_DEMO-easata.onrender.com-0D817A?style=for-the-badge&logo=render&logoColor=white" alt="Open the live EASATA demo"></a>
  <a href="https://easata.onrender.com/api/health"><img src="https://img.shields.io/website?url=https%3A%2F%2Feasata.onrender.com%2Fapi%2Fhealth&label=API&up_message=healthy&down_message=starting" alt="EASATA API health"></a>
  <img src="https://img.shields.io/badge/data-synthetic-2563EB?style=flat-square" alt="Synthetic data">
  <img src="https://img.shields.io/badge/outcomes-simulated-64748B?style=flat-square" alt="Simulated outcomes">
</p>

<p align="center">
  <a href="https://easata.onrender.com"><img src="docs/easata-globe.gif" alt="Looping EASATA globe animation. Open the live demo." width="720"></a>
</p>

> **Watch it in motion:** the landing page's globe and payment routes react as you scroll. [Launch the live experience](https://easata.onrender.com).
>
> **Synthetic data, simulated outcomes.** Trained on the public synthetic [Bank Transaction Fraud Detection Dataset](https://www.kaggle.com/datasets/nafiulislam490/bank-transaction-fraud-detection-dataset). No real bank, customer, or payment is involved. A score means "flagged for review," never proof of fraud. Freezes, holds, credits, and emails are simulated.

## System Map

```mermaid
flowchart LR
  subgraph train[Model training · Google Colab]
    data[Transaction data] --> features[Point-in-time features]
    features --> fit[CatBoost · calibration · threshold]
    fit --> bundle[Versioned model bundle]
  end

  subgraph app[EASATA application]
    web[React · Vite · Recharts] <-->|JSON API| api[FastAPI]
    api --> score[Scores · verified reasons · SHAP]
    api --> policy[Adaptive decision policy]
    api --> flow[Alerts · OTP · cases · audit trail]
    api <--> db[(SQLite demo state)]
    api --> monitor[PSI · alert-rate monitoring]
  end

  bundle -->|backend/model_bundle| api
  score --> policy --> web
  flow --> web

  classDef client fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
  classDef service fill:#dcfce7,stroke:#16a34a,color:#14532d
  classDef model fill:#fef3c7,stroke:#d97706,color:#78350f
  class web client
  class api,score,policy,flow,db,monitor service
  class data,features,fit,bundle model
```

## From Payment to Response

```mermaid
flowchart LR
  payment[Payment] --> signals[Point-in-time signals]
  signals --> probability[Calibrated fraud probability]
  signals --> anomaly[Anomaly percentile]
  probability --> loss[Expected loss = probability × amount]
  loss --> decision{Adaptive friction}
  anomaly -. separate signal .-> review[Analyst review]
  decision -->|PROCEED| pass[Continue]
  decision -->|STEP_UP| otp[Verify with OTP]
  decision -->|HOLD| ask[Ask if it was you]
  decision -->|BLOCK| stop[Stop payment]
  ask -->|Not me| freeze[Freeze account · open case]
  ask -->|It was me + OTP| trust[Remember a safe pattern]

  classDef signal fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
  classDef choice fill:#fef3c7,stroke:#d97706,color:#78350f
  classDef response fill:#dcfce7,stroke:#16a34a,color:#14532d
  class payment,signals,probability,anomaly,loss signal
  class decision choice
  class pass,otp,ask,stop,freeze,trust,review response
```

## Quick start

Requires **Python 3.10–3.12** (tested on 3.11). The model libraries must match the versions the notebook printed
(`backend/model_bundle/config.json → versions`); `requirements.txt` pins them.

**Bash (Linux/macOS)**
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put your bundle in place (skip if backend/model_bundle/ already has the files)
unzip -o /path/to/model_bundle.zip -d model_bundle
uvicorn app.main:app --port 8000            # first start loads 150k transactions (~30 s)
```

**PowerShell (Windows)**
```powershell
cd backend
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Expand-Archive -Force C:\path\to\model_bundle.zip model_bundle
uvicorn app.main:app --port 8000
```

Then open **http://localhost:8000/docs**. The bank-ops key is printed in the log and saved to `backend/data/ops_api_key.txt`
(or set `OPS_API_KEY`). Settings: copy `backend/.env.example` to `backend/.env`.

**Frontend** (Node 20+, second terminal)
```bash
cd frontend
npm install
npm run dev                                  # http://localhost:5173 (proxies /api to :8000 and adds the ops key)
```
`/` is the EASATA landing page (scroll-animated globe video). The console has Overview, Alerts, Simulate, Inbox, Cases and
Evaluation, and customers open `/respond?token=…` from the "Was this you?" email. `npm run build` writes `frontend/dist`,
which the API also serves at http://localhost:8000 after a restart (then paste the ops key via the 🔑 button).

| Task | Command (inside `backend/`, venv active) |
|---|---|
| Run API | `uvicorn app.main:app --port 8000` |
| Tests | `python -m pytest -q` |
| Full demo story (server running) | `python scripts/demo_flow.py` |
| Reset all workflow state | stop server, delete `data/app.db*`, start again |
| Optional real mail inbox | `docker run -p 8025:8025 -p 1025:1025 axllent/mailpit` → http://localhost:8025 |
| Retrain | open `notebook/fraud_catboost_colab.ipynb` in Colab (T4 GPU) → Run all → replace `model_bundle/` |

## Features

**Detection and explanation**
- CatBoost (Optuna-tuned, class-weighted), isotonic-calibrated probability, threshold chosen on validation.
- Isolation Forest anomaly **percentile** shown separately, never as a probability.
- Point-in-time per-customer features: amount vs usual, amount z-score, activity in 1h/24h/7d, unusual hour/day
  for *this* customer, first-time city/country/device/payment method/merchant, limited-history flag.
- **Verified reasons**: every reason is a template filled with row/baseline values; a verifier re-reads the data and
  checks every number in the text. Unverifiable reasons are dropped, and the API returns `numbers_verified`.
- **Two audiences**: analyst view (SHAP by feature and family, log-odds) and a **customer-safe message**
  (facts only, no scores or thresholds, so attackers learn nothing about the model).
- **What changed?** (replace a reason family with typical values → re-score) and a **minimal counterfactual**
  ("would not have been flagged if failed_attempts were 0"). Both are labelled as hypothetical, not causal.

**Response workflow (security-first)**
- **Adaptive friction**: PROCEED / STEP_UP (OTP) / HOLD ("Was this you?") / BLOCK, from probability and
  **expected loss = probability × amount**; the review queue is sorted by expected loss.
- **"Was this you?" email** with a single-use, 30-minute, 256-bit link token (only its SHA-256 is stored).
- **Opening the link changes nothing** (email scanners pre-open links); actions need a POST from a button.
- **Asymmetric trust**: "Not me" (freeze) is one click; "It was me" and unfreeze need an **OTP on another channel**,
  so an attacker who controls the email cannot clear the alert. OTPs are HMAC'd with a server secret, expire in
  5 minutes, and lock after 5 wrong tries.
- **Not me** in one atomic transaction: freeze (simulated), revoke sessions, require credential reset, hold the disputed
  payment, revoke trusted patterns, list related transactions, open a case, record feedback, email a confirmation.
  Double clicks are idempotent.
- **Case management** with RBI-framework-style deadlines: zero liability if reported within 3 working days of the
  bank's alert, shadow credit within 10 working days, resolution within 90 days. Countdowns, overdue flags, a validated
  state machine and an **append-only audit log** (enforced by SQLite triggers).
- **Trusted-pattern memory**: after an OTP-confirmed "It was me", similar payments (merchant, domestic/international,
  device) get one level less friction for 14 days. BLOCK is never lowered, patterns expire, and customers can revoke them.
- A **frozen account blocks** new (simulated) payments.

**Platform**
- Pending vs completed simulation: pending never writes anything; completed becomes history for later transactions.
- Idempotency keys, structured errors, request IDs, rate limits, `X-Ops-Key` for bank-ops endpoints,
  `Referrer-Policy: no-referrer`, `Cache-Control: no-store` on link pages.
- **Model health**: monthly alert rate and score, PSI drift per key feature.
- **Feedback loop**: `GET /api/ops/feedback.csv` exports customer-confirmed labels with features for retraining.
- Replay stream with cursor; demo inbox for emails/SMS.

## Measured results (test period Apr–Dec 2024, from `model_bundle/evaluation.json`)

| Model | Avg precision | ROC-AUC | Precision | Recall | Alerts / 1,000 |
|---|---|---|---|---|---|
| **CatBoost (calibrated)** | **0.121** | **0.728** | 0.136 | 0.351 | 141.9 |
| Isolation Forest | 0.070 | 0.574 | 0.077 | 0.269 | 191.1 |
| Amount-only baseline | 0.058 | 0.515 | 0.055 | 1.000 | 1000.0 |

### Model Comparison

```mermaid
xychart-beta
  title "Average precision · held-out test period"
  x-axis [CatBoost, Isolation Forest, Amount-only]
  y-axis "Average precision" 0 --> 0.13
  bar [0.121, 0.070, 0.058]
```

### Friction Mix

```mermaid
pie showData
  title Decision bands across the 150,000-payment demo set
  "Proceed · 65.3%" : 65.3
  "Ask for a code · 19.7%" : 19.7
  "Pause and ask · 13.6%" : 13.6
  "Stop · 1.4%" : 1.4
```

The fraud base rate is 5.5%, so random guessing gives an average precision of about 0.055. Calibration is good
(Brier 0.050; predicted ≈ observed per decile). Numbers come from the notebook run; retraining changes them, and the API
always serves the bundle's own `evaluation.json`.

**What the data allows (reported honestly)**
- Even the riskiest combination of signals in this dataset (night, 2+ failed attempts, international, PIN change)
  reaches only about 22% fraud, so no model can reach high precision here.
- Transaction amount carries no signal (AUC 0.515), and the six `fraud_type` values are statistically identical, so the
  explanation audit by fraud type is not meaningful.
- Per-customer habit features have low importance because the generator gives customers no stable habits. The features
  are verified against brute-force recomputation and leakage tests.
- PSI drift is "stable" across months, which is expected for a uniform synthetic generator.

## One-minute demo script

1. **Alerts** (sorted by expected loss): open the top alert. Show the band (BLOCK), the verified reasons with the ✓ badge,
   the SHAP families, the counterfactual ("if failed attempts were 0 → 0.09") and "What changed?".
2. Click **Send "Was this you?"**. Open the **Inbox**: plain, customer-safe wording, with no scores in it.
3. Open the link: nothing changes until a button is pressed. Click **Not me** → account **frozen (simulated)** → case opens
   with zero-liability status, shadow-credit and resolution deadlines, and the automatic actions timeline.
4. **Simulate** another payment from that customer: **BLOCK (account frozen)**.
5. Another customer: **It was me** asks for an OTP (on the phone, not the email) → trusted pattern created → the same
   pattern drops from HOLD to STEP_UP. Say: "BLOCK is never lowered."
6. **Cases**: move to INVESTIGATING → SHADOW_CREDITED; show the audit log. **Evaluation**: honest metrics and limitations.

(`python scripts/demo_flow.py` runs this story against the API and prints each step.)

## NVIDIA Nemotron on Nebius Token Factory

EASATA combines high-speed tabular fraud detection with generative agentic reasoning through a hybrid architecture:

```mermaid
flowchart LR
  subgraph detect[1. Detection · Tabular ML]
    txn[Transaction] --> catboost[CatBoost + TreeSHAP]
    catboost --> math[Calibrated risk + Log-odds]
  end

  subgraph guard[2. Guardrail · Truth Verifier]
    math --> verifier[reasons.verify & Leakage Guard]
  end

  subgraph reason[3. Reasoning · Nebius Token Factory]
    verifier <-->|OpenAI API| nemotron[NVIDIA Nemotron-70B
Serverless on H100 / L40S]
  end

  subgraph output[4. Verified Outputs]
    nemotron --> cust[Customer-Safe Alert]
    nemotron --> memo[Bank Forensic Memo]
  end
```

### Why NVIDIA Nemotron-70B?
- **Model:** `nvidia/Llama-3.1-Nemotron-70B-Instruct` (configurable via `NEBIUS_MODEL`).
- **High-Fidelity Reasoning:** Built for complex agentic workflows with leading benchmark scores in multi-step reasoning, mathematical accuracy, and instruction-following.
- **Negative Constraint Adherence:** In financial alerting, the model must **never reveal internal risk scores, thresholds, or SHAP log-odds** to the cardholder. Nemotron-70B excels at respecting strict negative guardrails where smaller models often hallucinate or leak internals.
- **Regulatory Case Memorandums:** Synthesizes point-in-time velocity, TreeSHAP contributions, and counterfactuals into formal audit memos compliant with the Reserve Bank of India (RBI) Zero-Liability framework (3-day customer notification deadline, 10-day provisional credit, 90-day resolution).

### Why Nebius Token Factory?
- **Zero-DevOps GPU Inference:** Provides managed, high-throughput serverless endpoints powered by NVIDIA H100 and L40S clusters.
- **Predictable Low Latency:** Delivers sub-second token latency crucial for real-time payment decisioning without managing complex Triton or vLLM deployments.
- **OpenAI-Compatible API:** Plug-and-play integration using standard HTTP clients (`httpx`).

### Truth Verifier: The Mathematical Guardrail
Generative LLMs risk hallucinating financial figures. EASATA eliminates this via `backend/app/reasons.py`:
1. **Leakage Pre/Post-Scan:** Rejects any customer text containing banned terms (`score`, `threshold`, `probability`, `shap`, `anomaly`, etc.) or raw probabilities.
2. **Deterministic Evidence Check:** Every numeric token in Nemotron's output is re-verified against ground-truth transaction evidence. If an invented number is detected, EASATA automatically rejects the output and falls back to deterministic templates (`source: "template"`).

### Endpoints
- `POST /api/alerts/{id}/nemotron/customer-message`: Customer alert rewrite with leakage & verifier guards.
- `POST /api/alerts/{id}/nemotron/case-memo`: Formal forensic case memorandum with RBI SLA alignment.

### Environment Variables & Setup
Set in `backend/.env` (or cloud secrets in `render.yaml`):
```bash
NEBIUS_API_KEY="your-nebius-token-factory-key"
NEBIUS_BASE_URL="https://api.studio.nebius.ai/v1"
NEBIUS_MODEL="nvidia/Llama-3.1-Nemotron-70B-Instruct"
# Optional Tavily API key for real-time merchant scam/reputation lookup:
TAVILY_API_KEY="tvly-your-key"
```

### Running the Live Evaluation Script
To evaluate Nemotron across 50 high-risk alerts:
```bash
cd backend
python scripts/eval_nemotron.py
```
*(If `NEBIUS_API_KEY` is not set, the script exits cleanly without fabricating numbers).*

## Tests (38)

`python -m pytest -q` checks:
- the bundle reproduces saved scores, and the alert flag matches the notebook;
- no label or ID columns are used as features;
- every number in explanations traces to data, and the verifier rejects invented numbers;
- customer messages carry no model internals;
- history features match brute force and don't change when future rows are added; cold-start flags work;
- viewing a link is read-only; tokens are never stored in plaintext; expired, invalid and used links are rejected;
- "Not me" is idempotent; the frozen account blocks payments;
- OTP is required, expires and locks after 5 tries; unfreeze needs an OTP;
- case transitions are valid, audit rows can't be changed, and working-day deadlines are correct;
- decision bands, trusted-pattern lowering, revocation and expiry work;
- pending simulation writes nothing; committed simulation updates history;
- idempotency keys, structured validation errors, rate limits and the dashboards work;
- Nemotron missing-key fallback, timeout fallback, leakage-guard rejection, verifier rejection of invented numbers, success path, case memo generation, and ops authorization work.

## Limitations

- **Synthetic data**: the realism is limited by the generator (see "What the data allows"). The label comes from the
  generator, not from real investigations.
- **Simulated outcomes**: no real payment is frozen, held or credited. Email/SMS go to an outbox (and optionally Mailpit).
- **History starts in the test period**: the bundle holds only Apr–Dec 2024, so simulated customers' in-app history
  starts there. Training-time history features came from the full dataset.
- **Deadlines** follow the RBI framework's structure but skip weekends only (no bank holidays). This is not legal advice.
- **Demo endpoints expose secrets on purpose**: `GET /api/demo/inbox` shows the emails and SMS, including link tokens and
  OTPs, so the demo works without a phone or mail server. Set `DEMO_MODE=0` anywhere that isn't a local demo. Dashboard
  read endpoints are unauthenticated in this demo.
- **Single-process demo**: the in-memory rate limiter and SQLite suit one server. Production would use PostgreSQL,
  a shared rate limiter, real authentication for analysts, and a real SMS/email provider.
- **Trusted patterns** are coarse (merchant category × domestic/international × device type).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
