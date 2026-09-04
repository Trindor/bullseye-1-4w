# Bullseye Supabase Setup

This directory is the canonical database setup for the Bullseye 1–4W application as of the validated **Phase 4R.5A** baseline.

## Why this exists

Before Phase 4S.2, Bullseye's database evolved through separate SQL snippets during development. The live application worked, but a fresh deployment could not recreate the database from the GitHub repository alone.

Phase 4S.2 fixes that reproducibility gap.

## Files

- `schema.sql` — canonical database objects required by Bullseye.
- `verify.sql` — read-only checks to confirm the required tables, RPC, RLS, privileges, and later-phase columns exist.

## Required durable objects

Bullseye currently expects six tables:

1. `bullseye_positions`
2. `bullseye_candidates`
3. `bullseye_closed_trades`
4. `bullseye_early_warning_snapshots`
5. `bullseye_app_settings`
6. `bullseye_candidate_outcomes`

It also expects the `bullseye_close_trade(...)` RPC for atomic close-and-archive behavior.

## Fresh deployment procedure

1. Create a Supabase project.
2. Open **SQL Editor**.
3. Paste and run `schema.sql`.
4. Paste and run `verify.sql`.
5. Confirm the verification results match the comments in `verify.sql`.
6. Add the following values to Streamlit secrets:
   - `SUPABASE_URL`
   - `SUPABASE_SECRET_KEY`
   - `BULLSEYE_OWNER_ID`
7. Never commit actual secret values to GitHub.

## Current access model

The validated Phase 4R.5A app sends the configured Supabase secret in the HTTP `apikey` header. The current database-facing role is `service_role`.

The Bullseye tables revoke access from `anon` and `authenticated`. The application is therefore dependent on keeping the Streamlit secret private and using server-side execution.

This is one reason owner isolation and public-app exposure remain Phase 4S hardening topics.

## Important Phase 4S.2 rule

**Do not run `schema.sql` against the current live Bullseye database merely because this file was added to GitHub.**

The production database already contains the working objects. During Phase 4S.2 the purpose is to make the setup reproducible and auditable. A fresh/test database is the appropriate place to prove full rebuild capability.

## What this phase does not change

- Bullseye 4.0 scoring
- 4R staging/readiness logic
- 4R.3 Opportunity State
- live position management
- candidate workflow
- existing production data
- Streamlit `app.py`

Phase 4S.2 is infrastructure/documentation hardening only.
