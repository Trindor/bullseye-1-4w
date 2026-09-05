# Phase 4S.4A — Durable Storage / Identity / Lot-Behavior Audit

## Scope

This is a documentation-only hardening phase. It does **not** change `app.py`, Bullseye 4.0 scoring, Supabase schema, market-data behavior, position-management math, or any live trade record.

Baseline reviewed: Phase 4S.3D — Historical Validation Failure Visibility.

## Executive finding

Bullseye's current durable-storage model is workable for one private owner and one active consolidated position per ticker, but several assumptions are implicit rather than enforced. Before future automation or wider access, those assumptions should be made explicit and then hardened.

## Findings

### 1. Owner isolation depends on one configured owner ID — HIGH

All durable reads/writes are scoped with `owner_id`, which is good. However, the app obtains that owner from the server-side `BULLSEYE_OWNER_ID` secret. There is no authenticated per-user identity in the current app.

**Consequence:** if the deployed Streamlit app is accessible to another person, that person is operating under the same configured owner identity and can potentially view or mutate the same Bullseye durable records through the UI.

**Current-safe assumption:** Bullseye is a private single-owner app.

**Future hardening direction:** preserve the current single-owner mode, but formally gate multi-user/public deployment behind authentication and per-user owner mapping.

### 2. Position identity is derived from ticker + entry price — HIGH

`_phase4q5_position_key(ticker, entry)` returns a key based on the ticker and entry rounded to four decimal places.

**Consequence:** if DCA or another fill changes the average entry, the derived position key changes. A save after the average-price change can create a second durable position row instead of updating the original trade identity.

This also makes entry price part of identity even though entry is a mutable property of an evolving position.

**Future hardening direction:** introduce an immutable trade/position ID created when the position is first opened. Keep ticker and average entry as data fields, not identity.

### 3. Held-position display collapses multiple rows to newest row per ticker — HIGH

`_phase4q6_list_held_positions()` requests all open durable position rows, sorts newest first, then keeps only the first row for each ticker.

**Consequence:** two legitimate simultaneous lots/trades in the same ticker cannot be represented independently in the Held Positions interface. An older open row can exist in storage while being hidden by the newer row.

**Future hardening direction:** decide explicitly whether Bullseye supports:
- one consolidated position per ticker, or
- multiple independent trade lots per ticker.

For the user's current workflow, one consolidated position per ticker is probably the simpler near-term model, but it should be enforced rather than merely assumed.

### 4. Delete and close actions inherit ticker+entry identity risk — HIGH

Delete and close/archive operations target the same derived `position_key`.

**Consequence:** if a DCA-adjusted entry creates a new key, the old row can become orphaned or a close action can target only one of several durable rows representing what the user thinks is one trade.

**Future hardening direction:** resolve after immutable position ID design.

### 5. Account size is durable but owner-wide — MEDIUM

`bullseye_app_settings` stores one account-size value per owner. This is appropriate for the current single-account Bullseye workflow.

**Consequence:** future support for multiple brokerage accounts would require account identity rather than only owner identity.

**Near-term action:** document current assumption; no code change needed now.

### 6. 4P caps portfolio risk but does not enforce total cash deployed — HIGH

Phase 4P reduces shares to stay inside total-risk and correlation-cluster risk caps. It does **not** reduce shares based on remaining cash/capital. Phase 4O separately warns when combined suggested position value exceeds account size or 75% of account value.

**Consequence:** a portfolio can satisfy the configured risk cap while the combined proposed capital deployment exceeds available account cash.

**Future hardening direction:** add a capital-deployment constraint to the same allocation pass that currently enforces total and cluster risk. This should be a later code phase with explicit regression tests.

### 7. Current 4P allocation is prospective, not aware of durable live-position cash/risk — MEDIUM/HIGH

The 4P planner sizes the currently evaluated actionable setups. It is not a full brokerage ledger and does not automatically subtract actual capital already tied up in durable held positions from available cash before sizing new opportunities.

**Consequence:** planner output can look safe in isolation while the real account already has exposure.

**Future hardening direction:** after storage identity is clarified, feed durable held-position exposure into portfolio capacity calculations.

## Recommended hardening order

1. **Define position identity model** — one consolidated position per ticker vs multiple lots.
2. **Add immutable position/trade ID** while preserving existing records.
3. **Make held-position listing consistent with the chosen identity model.**
4. **Add capital-deployment enforcement to 4P.**
5. **Incorporate durable live exposure into new-trade capacity.**
6. **Only then consider authenticated multi-user/public deployment.**

## What Phase 4S.4A deliberately does NOT do

- No live database migration.
- No change to existing COIN / CRM / NOW records.
- No change to closed-trade archive.
- No change to Bullseye 4.0.
- No change to Yahoo/yfinance calls.
- No change to Webull migration path.
- No change to Streamlit secrets.
- No attempt to solve multi-lot accounting in one step.

## Phase result

**Phase 4S.4A establishes the storage/identity/portfolio-capacity risk map.**

The highest-priority architectural issue is that mutable average entry price currently participates in durable position identity. The next code-hardening phase should solve that carefully without disturbing existing live trades.
