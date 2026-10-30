"""NVIDIA Nemotron integration on Nebius Token Factory / Nebius AI Studio.

Architectural Role:
1. Generative reasoning agent: transforms raw TreeSHAP and tabular metrics into human language.
2. Dual-audience synthesis:
   - Customer-safe alert rewrite: strictly empathetic, factual, with leakage guards ensuring
     zero disclosure of model internals (scores, thresholds, probabilities).
   - Analyst forensic investigation memo: structured case memo with TreeSHAP breakdown,
     counterfactuals, and RBI statutory SLA timelines (3-day zero liability, 10-day provisional credit).
3. The Truth Verifier Guarantee:
   Every generated text must pass reasons.verify() against underlying transaction evidence.
   Any invented number causes automatic rejection and fallback to deterministic templates.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

from . import clock
from .config import settings
from .reasons import verify

log = logging.getLogger("stdetector.nemotron")

# Forbidden terms that must never leak to customers
LEAKAGE_TERMS = [
    r"\bprobability\b",
    r"\bprobabilities\b",
    r"\bthreshold\b",
    r"\bthresholds\b",
    r"\bscore\b",
    r"\bscores\b",
    r"\bshap\b",
    r"\blog-odds\b",
    r"\bpercentile\b",
    r"\banomaly\b",
    r"\bcatboost\b",
    r"\bisolation\s+forest\b",
    r"\binternal\s+model\b",
    r"\bmodel\b",
    r"\balgorithm\b",
    r"\bbrier\b",
    r"\b0\.\d{2,}\b",  # raw floating point probabilities like 0.2536
]
_LEAKAGE_RE = re.compile("|".join(LEAKAGE_TERMS), re.IGNORECASE)


def check_leakage(text: str) -> tuple[bool, str | None]:
    """Returns (True, None) if customer-safe, or (False, reason) if internal terms leaked."""
    m = _LEAKAGE_RE.search(text)
    if m:
        return False, f"leakage_guard_triggered: found '{m.group(0)}'"
    return True, None


_cached_models: list[str] = []
_last_models_fetch: float = 0.0


def fetch_available_models(api_key: str | None = None) -> list[str]:
    """Fetch model list from Nebius Token Factory with fallback URL support."""
    global _cached_models, _last_models_fetch
    key = api_key or settings.nebius_api_key
    if not key:
        return []

    if _cached_models and (time.time() - _last_models_fetch < 300):
        return _cached_models

    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    urls = [
        f"{settings.nebius_base_url}/models",
        "https://api.tokenfactory.nebius.com/v1/models",
        "https://api.studio.nebius.ai/v1/models",
    ]
    seen_urls: list[str] = []
    for url in urls:
        if url in seen_urls:
            continue
        seen_urls.append(url)
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(url, headers=headers)
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    models = [m["id"] for m in data if isinstance(m, dict) and "id" in m]
                    if models:
                        _cached_models = models
                        _last_models_fetch = time.time()
                        log.info("Fetched %d models from %s", len(models), url)
                        return models
        except Exception as e:
            log.warning("Could not fetch models from %s: %s", url, e)
    return _cached_models


def get_nemotron_models(api_key: str | None = None) -> list[str]:
    """Return all available models matching 'nemotron'."""
    all_models = fetch_available_models(api_key)
    return [m for m in all_models if "nemotron" in m.lower()]


def resolve_model(api_key: str | None = None) -> tuple[str, bool]:
    """Resolve configured model to an available model, picking closest Nemotron if needed."""
    cfg = settings.nebius_model
    models = fetch_available_models(api_key)
    if not models:
        return cfg, False

    if cfg in models:
        return cfg, True

    nem_models = [m for m in models if "nemotron" in m.lower()]
    if nem_models:
        match_70b = [m for m in nem_models if "70b" in m.lower()]
        closest = match_70b[0] if match_70b else nem_models[0]
        log.warning("Configured model '%s' not in Nebius catalog. Resilient fallback resolved to: '%s'", cfg, closest)
        return closest, True

    return cfg, False


def call_nebius_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.2,
) -> tuple[str | None, float, dict, str | None]:
    """Call Nebius Token Factory OpenAI-compatible chat completions endpoint.

    Returns: (content, latency_ms, token_usage, error_msg)
    """
    api_key = settings.nebius_api_key
    if not api_key:
        return None, 0.0, {}, "missing_api_key"

    active_model, _ = resolve_model(api_key)
    url = f"{settings.nebius_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": active_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    t0 = time.perf_counter()
    attempts = 2
    last_err: Exception | None = None

    for attempt in range(attempts):
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                latency = round((time.perf_counter() - t0) * 1000, 2)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                tokens = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                log.info("Nemotron call succeeded in %.1fms (tokens: %s)", latency, tokens.get("total_tokens"))
                return content, latency, tokens, None
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
            last_err = e
            log.warning("Nebius API call attempt %d failed: %s", attempt + 1, e)
            if attempt == attempts - 1:
                break
            time.sleep(0.5)

    latency = round((time.perf_counter() - t0) * 1000, 2)
    return None, latency, {}, f"api_error: {last_err}"


def lookup_merchant_tavily(merchant_name: str) -> str | None:
    """Optional Tavily runtime search for merchant reputation/complaints."""
    api_key = settings.tavily_api_key
    if not api_key:
        return None

    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": f'"{merchant_name}" merchant scam fraud complaints reputation',
            "search_depth": "basic",
            "max_results": 2,
        }
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    snippet = results[0].get("content", "")[:200]
                    return f"Web intelligence ({merchant_name}): {snippet}..."
    except Exception as e:
        log.warning("Tavily search skipped: %s", e)
    return None


def generate_customer_message(
    row: dict,
    reasons_list: list[dict],
    base: dict,
    fallback_template: str,
    evidence: list[dict],
) -> dict[str, Any]:
    """Generate empathetic customer-facing notification with strict verifier and leakage guarantees."""
    system_prompt = (
        "You are EASATA Bank's Security Assistant. Write a concise, reassuring, and direct alert (2-3 sentences) "
        "to the cardholder regarding a suspicious transaction.\n"
        "STRICT MANDATORY RULES:\n"
        "1. NEVER mention risk scores, fraud probabilities, thresholds, SHAP values, weights, or algorithms.\n"
        "2. State only verified transaction facts provided. Do NOT invent numbers, dates, or currencies.\n"
        "3. Emphasize cardholder security and clarify that opening their alert link is safe."
    )

    user_prompt = f"""
Transaction Facts:
- Amount: {row.get('transaction_amount')}
- Merchant: {row.get('merchant_category')}
- Timestamp: {row.get('ts')}
- Device: {row.get('device_type')}
- Location: {row.get('city')}, {row.get('country')}
- Payment Method: {row.get('payment_method')}
- Customer-Safe Observations: {'; '.join(r['text'] for r in reasons_list if 'text' in r)}

Draft the customer alert:
"""

    content, latency, tokens, err = call_nebius_llm(system_prompt, user_prompt, max_tokens=250)

    # 1. API Call check
    if not content or err:
        return {
            "customer_message": fallback_template,
            "source": "template",
            "model": settings.nebius_model,
            "verified": True,
            "leakage_passed": True,
            "latency_ms": latency,
            "tokens": tokens,
            "fallback_reason": err or "missing_content",
        }

    # 2. Leakage Guard check
    leakage_passed, leak_err = check_leakage(content)
    if not leakage_passed:
        log.warning("Nemotron customer message rejected: %s", leak_err)
        return {
            "customer_message": fallback_template,
            "source": "template",
            "model": settings.nebius_model,
            "verified": True,
            "leakage_passed": False,
            "latency_ms": latency,
            "tokens": tokens,
            "fallback_reason": leak_err,
        }

    # 3. Truth Verifier check (must verify all numbers against evidence)
    is_verified = verify(content, evidence, row, base)
    if not is_verified:
        log.warning("Nemotron customer message failed verification: invented numbers detected.")
        return {
            "customer_message": fallback_template,
            "source": "template",
            "model": settings.nebius_model,
            "verified": False,
            "leakage_passed": True,
            "latency_ms": latency,
            "tokens": tokens,
            "fallback_reason": "verifier_failed_invented_numbers",
        }

    return {
        "customer_message": content,
        "source": "nemotron",
        "model": settings.nebius_model,
        "verified": True,
        "leakage_passed": True,
        "latency_ms": latency,
        "tokens": tokens,
        "fallback_reason": None,
    }


def generate_case_memo(
    row: dict,
    score: float,
    threshold: float,
    families: dict[str, float],
    counterfactual: dict,
    case: dict | None,
    actions: list[dict],
    base: dict,
) -> dict[str, Any]:
    """Generate structured bank operations forensic memorandum aligned with RBI Zero-Liability framework."""
    tavily_intel = lookup_merchant_tavily(str(row.get("merchant_category", "")))
    amt = float(row.get("transaction_amount", 0))

    # Evidence for verifier
    ev = [
        {"source": "row", "key": "transaction_amount", "value": amt, "scale": 1},
        {"source": "row", "key": "payment_method", "value": str(row.get("payment_method")), "scale": 1},
        {"source": "row", "key": "merchant_category", "value": str(row.get("merchant_category")), "scale": 1},
        {"source": "row", "key": "failed_attempts", "value": row.get("failed_attempts", 0), "scale": 1},
    ]

    fallback_memo = (
        f"INVESTIGATION MEMORANDUM\n"
        f"Transaction: {row.get('transaction_id')} | Amount: {amt:,.2f}\n"
        f"Calibrated Risk: {score:.1%} (Threshold: {threshold:.1%})\n"
        f"Primary Risk Drivers: {', '.join(f'{k} ({v:+.2f})' for k, v in families.items() if v > 0)}\n"
        f"Statutory Framework: RBI Guidelines (3-day zero liability, 10-day shadow credit, 90-day resolution).\n"
        f"Actions Taken: {len(actions)} recorded. Account frozen upon 'Not Me' report."
    )

    system_prompt = (
        "You are EASATA Bank's Senior Financial Crimes Investigator. "
        "Draft a structured, regulatory-grade Case Investigation Memorandum for an escalated suspicious transaction.\n"
        "Include 4 distinct sections:\n"
        "1. EXECUTIVE SUMMARY\n"
        "2. TREESHAP & VELOCITY FORENSICS\n"
        "3. STATUTORY LIABILITY (RBI Zero-Liability framework: 3 working days report = zero liability, "
        "10 working days provisional shadow credit, 90 days resolution)\n"
        "4. MANDATORY INVESTIGATOR ACTIONS\n"
        "STRICT RULE: Do NOT invent transaction amounts or false account details. Quote only provided numbers."
    )

    user_prompt = f"""
Case Data:
- Transaction ID: {row.get('transaction_id')}
- Customer ID: {row.get('customer_id')}
- Amount: {amt}
- Merchant: {row.get('merchant_category')} ({row.get('city')}, {row.get('country')})
- Calibrated Probability: {score:.4f} vs Alert Threshold {threshold:.4f}
- TreeSHAP Log-Odds Contributions: {families}
- Counterfactual Finding: {counterfactual.get('note', 'None')}
- External Intelligence: {tavily_intel or 'None available'}
- Case Status: {case.get('status') if case else 'Under Review'}
- Zero Liability Applicable: {'Yes' if (case and case.get('zero_liability')) else 'Under Assessment'}
- Automated System Actions: {[a.get('action') for a in actions]}
"""

    content, latency, tokens, err = call_nebius_llm(system_prompt, user_prompt, max_tokens=700)

    if not content or err:
        return {
            "case_memo": fallback_memo,
            "source": "template",
            "model": settings.nebius_model,
            "verified": True,
            "latency_ms": latency,
            "tokens": tokens,
            "tavily_intel": tavily_intel,
            "fallback_reason": err or "missing_content",
        }

    # Verify numbers in memo: must trace to transaction amount or statutory days
    is_verified = verify(content, ev, row, base)
    if not is_verified:
        log.warning("Nemotron case memo failed verification: invented numbers detected.")
        return {
            "case_memo": fallback_memo,
            "source": "template",
            "model": settings.nebius_model,
            "verified": False,
            "latency_ms": latency,
            "tokens": tokens,
            "tavily_intel": tavily_intel,
            "fallback_reason": "verifier_failed_invented_numbers",
        }

    return {
        "case_memo": content,
        "source": "nemotron",
        "model": settings.nebius_model,
        "verified": True,
        "latency_ms": latency,
        "tokens": tokens,
        "tavily_intel": tavily_intel,
        "fallback_reason": None,
    }


def record_generation(
    conn,
    transaction_id: str,
    generation_type: str,
    source: str,
    model: str,
    output_text: str,
    latency_ms: float,
    tokens: int,
    verified: bool,
    leakage_passed: bool,
    fallback_reason: str | None = None,
) -> None:
    """Save generation metadata to nemotron_generations table."""
    conn.execute(
        "INSERT INTO nemotron_generations (transaction_id, generation_type, source, model, output_text, "
        "latency_ms, tokens, verified, leakage_passed, fallback_reason, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            transaction_id,
            generation_type,
            source,
            model,
            output_text,
            latency_ms,
            tokens,
            int(verified),
            int(leakage_passed),
            fallback_reason,
            clock.iso(),
        ),
    )
