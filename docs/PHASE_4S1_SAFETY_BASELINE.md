# Bullseye Phase 4S.1 — Safety Baseline Audit

Baseline source: `app_phase4r5a_initial_opportunity_persistence_fix.py`

## Frozen checkpoint

Phase 4R.5A is the known-good functional baseline for Phase 4S hardening.
Phase 4S.1 intentionally changes **no Bullseye scoring, signal, position-management,
Supabase, scanner, or trading behavior**.

## Baseline facts

- Python source lines: 9,007
- Broad `except Exception` handlers: 54
- Broad handlers that can silently `continue`: 13
- Broad handlers that can silently `pass`: 2
- Python syntax/AST parse: PASS

## Highest-risk silent paths to address first

The following broad exception paths can discard a ticker/row without surfacing the reason:

- Line 1952
- Line 2000
- Line 2122
- Line 3318
- Line 3696
- Line 4559
- Line 6430
- Line 6462
- Line 6812
- Line 7019
- Line 7235
- Line 7404
- Line 7632

The following paths can silently `pass`:

- Line 8787
- Line 8956

Not every broad exception is automatically a bug. Some are deliberate graceful fallbacks
(e.g. unavailable Yahoo metadata). Phase 4S will distinguish **expected optional-data
fallbacks** from **critical workflow failures** before changing behavior.

## Critical workflows to protect with regression checks

1. Bullseye 4.0 scoring remains frozen.
2. Scanner produces rows without mutating scoring logic.
3. 4R stage/readiness and 4R.3 Opportunity State remain informational overlays.
4. Candidate save → Saved Candidates → Promote → Live save remains intact.
5. Live position management preserves protective-stop hierarchy and never loosens a stop automatically.
6. Close & Archive remains atomic and preserves closed-trade history.
7. 4R.2 scanner snapshots remain durable.
8. 4R.5 candidate outcome baseline is not overwritten; Initial Opportunity persists.
9. Supabase secret handling remains apikey-only for the current secret-key architecture.
10. Widget-bound session state is not mutated after widget instantiation.
11. Market cap remains informational only.
12. Broad / Focused / Custom universe selection does not alter Bullseye scoring.

## Phase 4S hardening order

- 4S.1: Safety baseline + regression guardrails.
- 4S.2: Supabase reproducibility (checked-in schema/RPC/grants documentation).
- 4S.3: Exception/logging hardening; remove dangerous silent failures.
- 4S.4: Storage identity / DCA / multiple-lot integrity audit.
- 4S.5: Portfolio deployed-cash enforcement and owner-data isolation review.
- 4S.6: Yahoo dependency boundary and request-pressure audit.
- 4S.7+: Gradual modularization only after guardrails are established.

## Deferred analytical audit

Score Sensitivity Audit: use NOW, T and controls to determine whether recent-day movement
is being over-counted across Bullseye components. This is an audit only; Bullseye 4.0
remains frozen unless evidence later justifies a separately validated scoring revision.

## UI footnote

Add restrained quick-glance icons to the dashboard Attention column during a safe UI-only
hardening step; no trading logic change.
