# Bullseye Phase 4S.3A — Silent Exception Classification

Baseline: validated Phase 4R.5A (`app.py` unchanged).

## Result

- Broad `except Exception` handlers inspected: **54**
- Silent `continue` / `pass` handlers requiring classification: **15**
- HIGH risk: **10**
- MEDIUM risk: **5**
- LOW/EXPECTED or REVIEW among silent paths: **0**

## Important finding

The most important silent paths are scanner/scoring and historical-validation loops. When `score_stock(...)` or a backtest observation raises an exception, the current code can `continue` without retaining the ticker/date/error. That can make a ticker disappear from results or reduce a validation sample without explaining why. This does **not** prove that any past missing ticker was caused by an exception; it establishes that the current code permits that failure mode.

## Silent-path inventory

### HIGH — line 1952 — `backtest_symbol`
- Action: continue
- Assessment: Can silently omit a ticker/observation when scoring fails.

### HIGH — line 2122 — `point_in_time_backtest_symbol`
- Action: continue
- Assessment: Can silently omit a ticker/observation when scoring fails.

### MEDIUM — line 2000 — `run_period_validation`
- Action: continue
- Assessment: Loop work can be skipped without preserving the failure reason.

### HIGH — line 3318 — `<module/UI>`
- Action: continue
- Assessment: Can silently omit a ticker/observation when scoring fails.

### MEDIUM — line 3696 — `<module/UI>`
- Action: continue
- Assessment: Loop work can be skipped without preserving the failure reason.

### MEDIUM — line 4559 — `<module/UI>`
- Action: continue
- Assessment: Loop work can be skipped without preserving the failure reason.

### HIGH — line 6462 — `<module/UI>`
- Action: continue
- Assessment: Can silently omit a ticker/observation when scoring fails.

### HIGH — line 6812 — `<module/UI>`
- Action: continue
- Assessment: Can silently omit a ticker/observation when scoring fails.

### HIGH — line 7019 — `<module/UI>`
- Action: continue
- Assessment: Can silently omit a ticker/observation when scoring fails.

### HIGH — line 7235 — `<module/UI>`
- Action: continue
- Assessment: Can silently omit a ticker/observation when scoring fails.

### HIGH — line 7404 — `<module/UI>`
- Action: continue
- Assessment: Can silently omit a ticker/observation when scoring fails.

### HIGH — line 7632 — `<module/UI>`
- Action: continue
- Assessment: Can silently omit a ticker/observation when scoring fails.

### HIGH — line 6430 — `<module/UI>`
- Action: continue
- Assessment: Can silently omit a ticker/observation when scoring fails.

### MEDIUM — line 8956 — `<module/UI>`
- Action: pass
- Assessment: Failure is suppressed completely; needs review before changing behavior.

### MEDIUM — line 8787 — `<module/UI>`
- Action: pass
- Assessment: Failure is suppressed completely; needs review before changing behavior.

## 4S.3B recommended repair strategy

Do **not** change scoring math. Do **not** make optional Yahoo metadata fatal.

1. Add a tiny diagnostic collector for scanner/scoring failures.
2. Replace scanner `except Exception: continue` with `except Exception as exc:` that records ticker + stage + exception type/message, then continues.
3. Show one compact warning after a scan only when failures occurred, with an expandable detail table.
4. Apply the same pattern to historical validation, recording ticker/date so failed samples are auditable.
5. Keep expected market-cap fallback as `NaN`, but count metadata failures separately rather than treating them as scanner failures.
6. Preserve existing user-visible error handlers and Supabase exceptions that already raise meaningful errors.
7. Re-run 4S.1 guardrails and require broad-exception count to stay at or below the baseline while silent critical paths decrease.

## No behavior change in 4S.3A

This phase is classification/audit only. The validated 4R.5A application remains unchanged.