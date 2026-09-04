"""Bullseye Phase 4S.1 source-level regression guardrails.

Run from repository root:
    python tests/test_phase4s1_guardrails.py

These tests deliberately do NOT import app.py because importing a Streamlit application
executes UI/runtime code. They provide a zero-dependency safety net for critical source
invariants while Phase 4S gradually introduces deeper unit tests.
"""
from pathlib import Path
import ast
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
SOURCE = APP.read_text(encoding="utf-8")

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def main():
    # 1. Entire application must remain syntactically valid.
    ast.parse(SOURCE)

    # 2. Current durable tables must remain distinct.
    for table in (
        "bullseye_positions",
        "bullseye_candidates",
        "bullseye_closed_trades",
        "bullseye_early_warning_snapshots",
        "bullseye_app_settings",
        "bullseye_candidate_outcomes",
    ):
        require(table in SOURCE, f"Missing durable table reference: {table}")

    # 3. Current Supabase secret architecture must retain apikey usage.
    require('"apikey"' in SOURCE or "'apikey'" in SOURCE,
            "Supabase apikey header reference is missing")

    # 4. Phase 4R.5 baseline field must remain present.
    require("initial_opportunity_state" in SOURCE,
            "4R.5 initial opportunity persistence field is missing")

    # 5. Market cap remains represented as context.
    require("Market Cap" in SOURCE, "Market Cap context is missing")

    # 6. Scan-universe controls remain available.
    for label in ("Broad", "Focused", "Custom"):
        require(label in SOURCE, f"Scan universe option missing: {label}")

    # 7. Candidate/live/closed lifecycle labels remain represented.
    for label in ("Candidate / Watching", "Entered / Live Position", "Closed Trade"):
        require(label in SOURCE, f"Position-state lifecycle label missing: {label}")

    # 8. Record current broad-exception count as a ratchet baseline.
    # Future hardening should reduce this count, not accidentally increase it.
    broad = len(re.findall(r"(?m)^\s*except Exception(?::|\s+as\s+)", SOURCE))
    require(broad <= 54,
            f"Broad exception-handler count increased above 4S.1 baseline: {broad} > 54")

    print("PASS — Bullseye Phase 4S.1 guardrails")
    print(f"app.py lines: {len(SOURCE.splitlines())}")
    print(f"broad Exception handlers: {broad}")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL — {exc}", file=sys.stderr)
        raise
