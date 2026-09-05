# Phase 4S.4B — 4S.4C Regression Checklist

This checklist is committed before implementation so Phase 4S.4C has an explicit pass/fail contract.

## Pre-migration production checks

- [ ] Record current live-position row count.
- [ ] Confirm COIN exists.
- [ ] Confirm CRM exists.
- [ ] Confirm NOW exists.
- [ ] Record each live trade's entry, remaining shares, current stop, highest R/state, notes, and realized P/L.
- [ ] Record current closed-trade row count.
- [ ] Record current candidate row count.

## Schema migration checks

- [ ] Add nullable `position_id` to live positions.
- [ ] Add nullable `position_id` to closed trades.
- [ ] Migration is additive only.
- [ ] No existing row deleted.
- [ ] No existing row count changes.
- [ ] Backfill assigns exactly one ID per legacy live row.
- [ ] Re-running backfill does not change already assigned IDs.

## Application compatibility checks

- [ ] Existing legacy rows still load.
- [ ] New rows receive immutable IDs.
- [ ] Existing rows are updated by immutable ID after backfill.
- [ ] Legacy position-key fallback remains available during transition.
- [ ] Held Positions still loads normally.
- [ ] Candidate -> Live promotion still works.
- [ ] Delete-test-position flow still targets the intended position.
- [ ] Close/archive remains atomic.

## Identity behavior tests

### DCA
- [ ] Add shares to a test position.
- [ ] Average entry changes.
- [ ] `position_id` does not change.
- [ ] No second active durable row is created.

### Partial exit
- [ ] Sell part of a test position.
- [ ] Remaining shares decrease.
- [ ] Realized P/L updates.
- [ ] `position_id` does not change.
- [ ] Position remains live.

### Full close
- [ ] Close a test position.
- [ ] Exactly one closed-trade record is created.
- [ ] Closed record preserves the same `position_id`.
- [ ] Live row is no longer active.
- [ ] Historical notes/management fields are preserved.

### Re-entry
- [ ] Re-enter the same ticker after prior trade is closed.
- [ ] New trade receives a different `position_id`.
- [ ] Prior closed trade remains unchanged.

## Frozen-logic checks

- [ ] Bullseye 4.0 scoring unchanged.
- [ ] 4R.3 opportunity-state logic unchanged.
- [ ] 4R.4 profit-protection observation unchanged.
- [ ] 4Q.2 management math unchanged.
- [ ] No additional Yahoo/yfinance call sites.
- [ ] No Webull-migration dependency added.

## Live validation

- [ ] COIN intact after deployment.
- [ ] CRM intact after deployment.
- [ ] NOW intact after deployment.
- [ ] Closed Trades history intact.
- [ ] Candidate Watchlist intact.
- [ ] No unexpected Supabase diagnostic errors.
- [ ] Normal scanner startup remains clean.

## Exit criterion

Phase 4S.4C is not considered complete until the immutable identity path is proven on test/simulated data **and** the existing production records remain intact.
