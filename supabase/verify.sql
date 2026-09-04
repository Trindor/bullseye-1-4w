-- Bullseye 1–4W
-- Phase 4S.2 — Read-only database verification
--
-- Run AFTER schema.sql on a fresh/test Supabase project.
-- This does not insert, update, or delete Bullseye data.

-- A. Required tables
select table_name
from information_schema.tables
where table_schema = 'public'
  and table_name in (
    'bullseye_positions',
    'bullseye_candidates',
    'bullseye_closed_trades',
    'bullseye_early_warning_snapshots',
    'bullseye_app_settings',
    'bullseye_candidate_outcomes'
  )
order by table_name;

-- Expected: 6 rows.


-- B. Required close-trade RPC
select routine_name
from information_schema.routines
where routine_schema = 'public'
  and routine_name = 'bullseye_close_trade';

-- Expected: bullseye_close_trade


-- C. RLS enabled on all six Bullseye tables
select
    c.relname as table_name,
    c.relrowsecurity as rls_enabled
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public'
  and c.relname in (
    'bullseye_positions',
    'bullseye_candidates',
    'bullseye_closed_trades',
    'bullseye_early_warning_snapshots',
    'bullseye_app_settings',
    'bullseye_candidate_outcomes'
  )
order by c.relname;

-- Expected: rls_enabled = true for all six.


-- D. Table privileges
select
    table_name,
    grantee,
    string_agg(privilege_type, ', ' order by privilege_type) as privileges
from information_schema.role_table_grants
where table_schema = 'public'
  and table_name in (
    'bullseye_positions',
    'bullseye_candidates',
    'bullseye_closed_trades',
    'bullseye_early_warning_snapshots',
    'bullseye_app_settings',
    'bullseye_candidate_outcomes'
  )
  and grantee in ('anon', 'authenticated', 'service_role')
group by table_name, grantee
order by table_name, grantee;

-- Expected:
--   service_role appears with the required privileges.
--   anon/authenticated should not have Bullseye table privileges.


-- E. 4R.3 snapshot opportunity columns
select column_name
from information_schema.columns
where table_schema = 'public'
  and table_name = 'bullseye_early_warning_snapshots'
  and column_name in (
    'opportunity_state',
    'opportunity_icon',
    'opportunity_rank',
    'opportunity_why',
    'opportunity_next_trigger'
  )
order by column_name;

-- Expected: 5 rows.


-- F. 4R.5A baseline field
select column_name
from information_schema.columns
where table_schema = 'public'
  and table_name = 'bullseye_candidate_outcomes'
  and column_name = 'initial_opportunity_state';

-- Expected: initial_opportunity_state
