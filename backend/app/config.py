"""Runtime settings. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env support (KEY=VALUE lines). Real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(BACKEND_DIR / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


class Settings:
    def __init__(self) -> None:
        self.bundle_dir = Path(_env("BUNDLE_DIR", str(BACKEND_DIR / "model_bundle")))
        self.db_path = Path(_env("DB_PATH", str(BACKEND_DIR / "data" / "app.db")))
        self.frontend_dist = Path(_env("FRONTEND_DIST", str(PROJECT_DIR / "frontend" / "dist")))

        # Load only part of the demo data (evenly spaced over time). 0 = all rows. Used by tests.
        self.demo_max_rows = int(_env("DEMO_MAX_ROWS", "0"))

        # Links inside "Was this you?" emails point here (the frontend's /respond page).
        self.public_url = _env("PUBLIC_URL", "http://localhost:5173").rstrip("/")
        self.cors_origins = [o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

        # Demo delivery: every email/SMS is stored in the outbox table (GET /api/demo/inbox).
        # If an SMTP server is reachable (e.g. Mailpit on localhost:1025) emails are also sent there.
        self.demo_mode = _env("DEMO_MODE", "1") == "1"
        self.demo_email = _env("DEMO_EMAIL", "customer@example.test")
        self.demo_phone = _env("DEMO_PHONE", "+91-90000-00000")
        self.smtp_enabled = _env("SMTP_ENABLED", "1") == "1"
        self.smtp_host = _env("SMTP_HOST", "localhost")
        self.smtp_port = int(_env("SMTP_PORT", "1025"))
        self.smtp_user = _env("SMTP_USER", "")
        self.smtp_password = _env("SMTP_PASSWORD", "")
        self.smtp_starttls = _env("SMTP_STARTTLS", "0") == "1"
        self.mail_from = _env("MAIL_FROM", "alerts@stdetector.demo")

        # Bank-operations endpoints (/api/ops/*) require the X-Ops-Key header.
        # If OPS_API_KEY is not set, a random key is generated at first start and saved next to the DB.
        self.ops_api_key = _env("OPS_API_KEY", "")

        self.token_ttl_min = int(_env("TOKEN_TTL_MIN", "30"))
        self.otp_ttl_min = int(_env("OTP_TTL_MIN", "5"))
        self.otp_max_attempts = int(_env("OTP_MAX_ATTEMPTS", "5"))
        self.rate_limit_per_min = int(_env("RATE_LIMIT_PER_MIN", "30"))
        self.trusted_pattern_days = int(_env("TRUSTED_PATTERN_DAYS", "14"))

        # Case deadlines modelled on the RBI framework for unauthorised electronic banking transactions:
        # report within 3 working days of the bank's alert -> zero customer liability; shadow credit within
        # 10 working days of the report; resolution within 90 days. Weekends skipped; bank holidays are not.
        self.zero_liability_working_days = 3
        self.shadow_credit_working_days = 10
        self.resolution_days = 90

        # Adaptive friction. Empty = derived from the bundle at startup (see decision.py).
        self.expected_loss_high = float(_env("EXPECTED_LOSS_HIGH", "0") or 0)
        self.nebius_api_key = _env("NEBIUS_API_KEY", "")
        self.nebius_base_url = _env("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1").rstrip("/")
        self.nebius_model = _env("NEBIUS_MODEL", "nvidia/Llama-3.1-Nemotron-70B-Instruct")
        self.tavily_api_key = _env("TAVILY_API_KEY", "")

    def resolve_ops_key(self) -> str:
        if self.ops_api_key:
            return self.ops_api_key
        key_file = self.db_path.parent / "ops_api_key.txt"
        key_file.parent.mkdir(parents=True, exist_ok=True)
        if key_file.exists():
            self.ops_api_key = key_file.read_text().strip()
        else:
            self.ops_api_key = secrets.token_urlsafe(24)
            key_file.write_text(self.ops_api_key)
            try:
                os.chmod(key_file, 0o600)
            except OSError:
                pass
        return self.ops_api_key


settings = Settings()
