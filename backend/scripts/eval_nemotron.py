"""Evaluation script for NVIDIA Nemotron-70B on Nebius Token Factory.

Evaluates N=50 alerts from demo data:
1. Measures live latency (p50, p95)
2. Checks truth verifier pass rate
3. Checks leakage guard pass rate
4. Computes template fallback rate and token usage
5. Saves results to backend/data/nemotron_eval.json if run live

If NEBIUS_API_KEY is not set, prints instructions and exits cleanly (no fabricated output).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import pandas as pd

from app import nemotron, reasons
from app.config import settings
from app.db import Database
from app.model_service import ModelService


def main():
    print("=" * 80)
    print("EASATA: NVIDIA Nemotron-70B on Nebius Token Factory Evaluation")
    print("=" * 80)

    api_key = os.getenv("NEBIUS_API_KEY") or settings.nebius_api_key
    if not api_key:
        print("\n[!] NEBIUS_API_KEY is not set.")
        print("To run live evaluation against Nebius Token Factory:")
        print("  1. Obtain your API key from https://studio.nebius.ai/")
        print("  2. Set the environment variable: export NEBIUS_API_KEY='your-key'")
        print("     or add NEBIUS_API_KEY=your-key to backend/.env")
        print("  3. Run: python scripts/eval_nemotron.py")
        print("\nExiting cleanly without fabricating numbers.")
        sys.exit(0)

    print(f"\n[+] Connecting to Nebius Token Factory: {settings.nebius_base_url}")
    print(f"[+] Model: {settings.nebius_model}")

    # Load transactions and model service
    model = ModelService(settings.bundle_dir)
    db = Database(settings.db_path)
    if not db.transactions_loaded():
        print("[+] Loading transactions into demo DB...")
        db.load_transactions(settings.bundle_dir / "demo_transactions.csv.gz", settings.demo_max_rows)

    with db.read() as conn:
        alerts = pd.read_sql("SELECT * FROM transactions WHERE alert=1 ORDER BY score DESC LIMIT 50", conn)

    n_samples = len(alerts)
    print(f"[+] Running evaluation on {n_samples} high-risk alerts...\n")

    latencies: list[float] = []
    tokens_list: list[int] = []
    verified_count = 0
    leakage_passed_count = 0
    nemotron_success_count = 0

    for idx, row_dict in enumerate(alerts.to_dict("records"), 1):
        expl = model.explain(row_dict)
        raw_reasons = expl.get("reasons", [])
        base = model.baseline
        cust_msg_obj = reasons.customer_message(row_dict, raw_reasons, base)

        t0 = time.perf_counter()
        res = nemotron.generate_customer_message(
            row=row_dict,
            reasons_list=raw_reasons,
            base=base,
            fallback_template=cust_msg_obj.get("text", expl.get("customer_message", "")),
            evidence=cust_msg_obj.get("evidence", []),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        latencies.append(res.get("latency_ms") or elapsed_ms)
        if res.get("tokens") and res["tokens"].get("total_tokens"):
            tokens_list.append(res["tokens"]["total_tokens"])

        if res.get("verified"):
            verified_count += 1
        if res.get("leakage_passed"):
            leakage_passed_count += 1
        if res.get("source") == "nemotron":
            nemotron_success_count += 1

        print(f"[{idx:02d}/{n_samples:02d}] Txn {row_dict['transaction_id']} | Source: {res['source']:<8} | "
              f"Verified: {str(res['verified']):<5} | Leakage passed: {str(res['leakage_passed']):<5} | "
              f"Latency: {res['latency_ms']:.0f}ms")

    lat_arr = np.array(latencies)
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model": settings.nebius_model,
        "endpoint": settings.nebius_base_url,
        "n_evaluated": n_samples,
        "latency_p50_ms": round(float(np.percentile(lat_arr, 50)), 1),
        "latency_p95_ms": round(float(np.percentile(lat_arr, 95)), 1),
        "latency_mean_ms": round(float(np.mean(lat_arr)), 1),
        "verifier_pass_rate": round(verified_count / n_samples, 4),
        "leakage_pass_rate": round(leakage_passed_count / n_samples, 4),
        "nemotron_generation_rate": round(nemotron_success_count / n_samples, 4),
        "template_fallback_rate": round(1.0 - (nemotron_success_count / n_samples), 4),
        "avg_tokens_per_call": round(float(np.mean(tokens_list)), 1) if tokens_list else None,
    }

    out_path = BACKEND_DIR / "data" / "nemotron_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    for k, v in results.items():
        print(f"  {k:<28}: {v}")
    print(f"\n[+] Results saved to {out_path}")


if __name__ == "__main__":
    main()
