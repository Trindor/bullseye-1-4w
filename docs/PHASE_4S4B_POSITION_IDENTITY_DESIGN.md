# Phase 4S.4B — Position Identity Design & Migration Guardrails

## Purpose

Phase 4S.4B defines the durable identity contract for Bullseye positions **before any production code or database migration is attempted**.

This phase is documentation/guardrail only.

It does **not** change:

- `app.py`
- Bullseye 4.0 scoring
- live-position math
- Supabase production schema
- existing live trades
- closed-trade history
- Yahoo/yfinance behavior
- Webull migration architecture

The current production assumption remains: **one private Bullseye owner, with one consolidated active position per ticker.**

## Problem being solved

Today, Bullseye's durable live-position identity is derived from mutable trade data:

> ticker + entry price

That is unsafe as a permanent identifier because an average entry can change after DCA or another fill.

A durable trade identity must survive DCA, partial exits, protective-stop changes, position-mark updates, score changes, breakout/invalidation changes, app restarts, and future data-source migration.

## Chosen near-term model

### One consolidated active position per ticker

Bullseye will continue to treat one ticker as one consolidated active swing position for the current 1–4 week workflow.

This intentionally avoids introducing full tax-lot accounting during hardening.

Examples:

- Existing CRM position = one active CRM trade record.
- Buying more CRM updates the same durable trade identity and may change average entry.
- Partial CRM exits reduce remaining shares but do not create a new active identity.
- Once CRM is fully closed and archived, a later new CRM trade receives a **new immutable ID**.

Multiple simultaneous independent active CRM trades are **out of scope** for this hardening cycle.

## New identity contract

### Immutable `position_id`

Every active Bullseye position will ultimately have an immutable `position_id`.

Properties:

1. Created once when a new live position is first saved.
2. Never changes during the life of that trade.
3. DCA does not change it.
4. Partial exits do not change it.
5. Closing/archive preserves the same ID in the closed-trade record.
6. Re-entering the same ticker after the old trade is closed creates a new ID.
7. The ID is not based on ticker, price, date, score, or any other mutable field.

Recommended format:

`pos_<UUID>`

The exact UUID-generation implementation belongs to Phase 4S.4C.

## Legacy compatibility contract

Existing production rows do not currently have a `position_id`.

Phase 4S.4C must therefore support a transition period.

Required behavior:

- Existing rows remain readable.
- Existing COIN / CRM / NOW live positions must remain intact.
- Existing closed trades remain intact.
- Legacy `position_key` continues to work as a temporary lookup fallback.
- New code must prefer `position_id` when present.
- A migration/backfill must be idempotent.
- Re-running the migration must not create new identities for rows already backfilled.
- No live record may be deleted merely because it lacks `position_id`.

## Proposed lookup hierarchy

During transition:

1. If `position_id` exists, use it as the primary durable identity.
2. Otherwise, use the legacy `position_key`.
3. Once a legacy row is backfilled, all future updates target `position_id`.
4. `position_key` may remain stored for audit/history during the transition.

## DCA behavior after migration

When shares are added to an existing active ticker:

- keep the same `position_id`
- update initial/remaining shares as appropriate
- update average entry according to the existing accounting method
- preserve original trade identity
- preserve historical ratchet/protection state unless intentionally reset by an explicit future rule
- do not create a second live row merely because average entry changed

This is the primary defect the immutable ID is intended to prevent.

## Partial exit behavior after migration

A partial exit:

- keeps the same `position_id`
- reduces remaining shares
- updates realized P/L
- leaves the position live while remaining shares > 0
- preserves the same trade identity

## Full close behavior after migration

When remaining shares reach zero and the user closes/archives the trade:

- archive using the same `position_id`
- preserve legacy `position_key` if available
- remove/close the live row using `position_id`
- create exactly one closed-trade record
- preserve notes, entry, exit, realized P/L, original stop, highest R/state, and exit reason

The close operation must remain atomic.

## Re-entry behavior

If a ticker has no active live position and the user later enters it again:

- generate a new `position_id`
- do not reuse the prior trade's ID
- do not overwrite the old closed trade

Example:

`COIN trade A -> pos_111... -> closed`

Later:

`COIN trade B -> pos_222... -> live`

## Owner isolation

`position_id` does not replace owner scoping.

Durable records must still be scoped by `owner_id`.

The effective durable key becomes conceptually:

> owner_id + position_id

Bullseye remains a private single-owner app for now. Authenticated multi-user identity remains a future hardening item.

## Database design direction for Phase 4S.4C

Likely additive schema change:

- add nullable `position_id` to `bullseye_positions`
- add nullable `position_id` to `bullseye_closed_trades`
- backfill existing live rows once
- preserve existing legacy keys
- add appropriate uniqueness/indexing only after the backfill is verified

Important: **do not make `position_id` NOT NULL in the same first migration step.**

A staged migration reduces risk to existing production data.

## Required 4S.4C safety sequence

1. Take/read current production row inventory.
2. Verify the three current live positions are present before migration.
3. Add nullable ID columns only.
4. Backfill IDs idempotently.
5. Verify row counts are unchanged.
6. Verify COIN / CRM / NOW retain all existing values.
7. Add code that reads `position_id` first and legacy key second.
8. Test DCA identity in simulation/test data.
9. Test partial exit identity.
10. Test close/archive identity.
11. Only after live validation consider stronger database constraints.

## Regression guardrails

Phase 4S.4C must prove:

- no Bullseye 4.0 scoring change
- no additional Yahoo/yfinance calls
- no change to 4R.3 opportunity logic
- no change to 4R.4 profit-protection observation
- no change to 4Q.2 management math
- no loss of candidate records
- no loss of live records
- no loss of closed-trade records
- no duplicate active row created by a DCA average-entry change
- close/archive still creates exactly one closed record
- app remains compatible with legacy rows during transition

## Explicitly out of scope

Phase 4S.4B does not design brokerage tax-lot accounting, FIFO/LIFO, multiple simultaneous active lots per ticker, wash-sale accounting, automated trade execution, Webull order IDs, multi-user authentication, dynamic universe logic, or new scoring features.

## Phase result

**Phase 4S.4B freezes the intended position-identity contract before implementation.**

The implementation target is now clear:

> A Bullseye live trade gets one immutable `position_id` for its entire lifecycle. Mutable fields such as average entry may change without changing the trade's durable identity.
