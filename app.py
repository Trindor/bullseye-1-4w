import math
import traceback
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Bullseye 1–4W", layout="wide")

st.title("🎯 Bullseye 1–4W")
st.caption("Phase 4R.5 — Candidate Outcome Journal; observed candidate outcomes are measured for forward validation only and do not change Bullseye scoring, 4R.3 Opportunity State, or Phase 4Q management.")

DEFAULT_TICKERS = """
AAPL MSFT NVDA AMZN META GOOGL AVGO AMD TSLA NFLX
JPM V MA XOM CVX COST WMT ORCL CRM PLTR MU INTC
QCOM AMAT LRCX MRVL PANW CRWD UBER HOOD COIN
LLY UNH JNJ ABBV ISRG BSX ABT MDT SYK
""".split()


BROAD_TICKERS = """
AAPL MSFT NVDA AMZN META GOOGL AVGO AMD TSLA NFLX ORCL CRM ADBE NOW IBM
QCOM AMAT LRCX MU MRVL INTC TXN ADI KLAC MCHP PANW CRWD FTNT DDOG SNOW
JPM BAC WFC C GS MS BLK SCHW AXP V MA PYPL COF USB PNC TFC BK
XOM CVX COP EOG SLB OXY MPC PSX VLO KMI WMB HAL
LLY UNH JNJ ABBV MRK PFE TMO DHR ABT MDT SYK BSX ISRG GILD AMGN BMY
WMT COST TGT HD LOW NKE SBUX MCD CMG TJX ROST BKNG MAR
CAT DE GE RTX LMT NOC ETN HON UPS FDX UNP CSX WM EMR PH
PG KO PEP CL KMB PM MO EL MDLZ GIS KHC
DIS CMCSA T TMUS VZ NFLX SPOT WBD
NEE DUK SO AEP EXC SRE XEL ED D
LIN APD SHW ECL FCX NUE NEM DOW DD
AMT PLD EQIX SPG O CCI WELL PSA
UBER ABNB DASH HOOD COIN PLTR SHOP MELI RBLX
""".split()

@st.cache_data(ttl=900)
def download_prices(tickers):
    return yf.download(
        tickers=tickers,
        period="5y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )


def _format_market_cap(value):
    """Compact display only; underlying value remains raw dollars."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value) or value <= 0:
        return "—"
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def _fetch_one_market_cap(ticker):
    """Fetch Yahoo/yfinance market cap without changing any Bullseye determination."""
    try:
        cap = yf.Ticker(str(ticker).upper()).fast_info.market_cap
        cap = float(cap) if cap is not None else np.nan
        return cap if np.isfinite(cap) and cap > 0 else np.nan
    except Exception:
        return np.nan


@st.cache_data(ttl=21600, show_spinner=False)
def get_market_caps(tickers_tuple):
    """Cache market-cap metadata for six hours; fetch concurrently for broad scans."""
    symbols = tuple(sorted(set(str(t).upper().strip() for t in tickers_tuple if str(t).strip())))
    caps = {t: np.nan for t in symbols}
    if not symbols:
        return caps
    with ThreadPoolExecutor(max_workers=min(12, len(symbols))) as pool:
        futures = {pool.submit(_fetch_one_market_cap, t): t for t in symbols}
        for future in as_completed(futures):
            t = futures[future]
            try:
                caps[t] = future.result()
            except Exception:
                caps[t] = np.nan
    return caps


def get_market_cap(ticker):
    return get_market_caps((str(ticker).upper().strip(),)).get(str(ticker).upper().strip(), np.nan)


def get_position_mark(ticker):
    """Return latest extended-hours-aware mark for actual-position accounting only."""
    result = {
        "price": np.nan,
        "timestamp": None,
        "session": "Unavailable",
        "source": "Yahoo/yfinance extended-hours intraday",
        "status": "No intraday mark available",
    }
    try:
        hist = yf.Ticker(ticker).history(
            period="5d",
            interval="1m",
            prepost=True,
            auto_adjust=False,
            actions=False,
        )
        if hist is None or hist.empty or "Close" not in hist.columns:
            return result

        closes = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if closes.empty:
            return result

        ts = pd.Timestamp(closes.index[-1])
        price = float(closes.iloc[-1])

        if ts.tzinfo is None:
            ts_et = ts.tz_localize("America/New_York")
        else:
            ts_et = ts.tz_convert("America/New_York")

        t = ts_et.time()
        if ts_et.weekday() >= 5:
            session = "Latest available / market closed"
        elif pd.Timestamp("04:00").time() <= t < pd.Timestamp("09:30").time():
            session = "Pre-market"
        elif pd.Timestamp("09:30").time() <= t < pd.Timestamp("16:00").time():
            session = "Regular session"
        elif pd.Timestamp("16:00").time() <= t <= pd.Timestamp("20:00").time():
            session = "After-hours"
        else:
            session = "Latest available / market closed"

        result.update({
            "price": price,
            "timestamp": ts_et,
            "session": session,
            "status": "OK",
        })
        return result
    except Exception as exc:
        result["status"] = f"Position mark fallback required: {exc}"
        return result





def _phase4q5_storage_config():
    """
    Read server-side Supabase settings from Streamlit secrets.

    Supports either:
      SUPABASE_URL = "https://PROJECT.supabase.co"
      SUPABASE_SECRET_KEY = "sb_secret_..."
      BULLSEYE_OWNER_ID = "bullseye_primary"

    or the grouped [bullseye_storage] format.
    """
    try:
        url = str(st.secrets.get("SUPABASE_URL", "")).rstrip("/")
        key = str(st.secrets.get("SUPABASE_SECRET_KEY", ""))
        owner_id = str(st.secrets.get("BULLSEYE_OWNER_ID", ""))

        if not (url and key and owner_id):
            cfg = st.secrets.get("bullseye_storage", {})
            url = url or str(cfg.get("supabase_url", "")).rstrip("/")
            key = key or str(cfg.get("supabase_secret_key", ""))
            owner_id = owner_id or str(cfg.get("owner_id", ""))

        configured = bool(url and key and owner_id)
        return {
            "configured": configured,
            "url": url,
            "key": key,
            "owner_id": owner_id,
        }
    except Exception:
        return {"configured": False, "url": "", "key": "", "owner_id": ""}


def _phase4q5_request(method, table, params=None, payload=None, prefer=None):
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        raise RuntimeError("Phase 4Q.5 durable storage is not configured.")

    query = urllib_parse.urlencode(params or {}, safe="(),.*")
    endpoint = f'{cfg["url"]}/rest/v1/{table}'
    if query:
        endpoint += "?" + query

    headers = {
        "apikey": cfg["key"],
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer

    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(endpoint, data=body, headers=headers, method=method)

    try:
        with urllib_request.urlopen(req, timeout=12) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return None
            return json.loads(raw)
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Durable storage HTTP {exc.code}: {detail[:500]}") from exc
    except Exception as exc:
        raise RuntimeError(f"Durable storage request failed: {exc}") from exc


def _phase4r2f_load_account_size(default=25000.0):
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        return float(default)

    params = {
        "select": "account_size",
        "owner_id": f"eq.{cfg['owner_id']}",
        "limit": "1",
    }
    rows = _phase4q5_request(
        "GET",
        "bullseye_app_settings",
        params=params,
    ) or []
    if not rows:
        return float(default)

    try:
        value = float(rows[0].get("account_size", default))
        return value if np.isfinite(value) and value >= 0 else float(default)
    except Exception:
        return float(default)


def _phase4r2f_save_account_size(account_size):
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        raise RuntimeError("Durable storage is not configured.")

    account_size = float(account_size)
    if not np.isfinite(account_size) or account_size < 0:
        raise ValueError("Account size must be a valid non-negative number.")

    params = {"on_conflict": "owner_id"}
    payload = {
        "owner_id": cfg["owner_id"],
        "account_size": account_size,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    _phase4q5_request(
        "POST",
        "bullseye_app_settings",
        params=params,
        payload=payload,
        prefer="resolution=merge-duplicates,return=minimal",
    )
    return account_size


def _phase4q5_position_key(ticker, entry):
    return f"{str(ticker).upper().strip()}|{float(entry):.4f}"


def _phase4q5_save_position(
    ticker,
    position_state,
    entry,
    initial_shares,
    remaining_shares,
    realized_pl,
    original_stop,
    current_stop_input,
    live_state,
):
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        return {"ok": False, "status": "not_configured"}

    position_key = _phase4q5_position_key(ticker, entry)
    payload = {
        "owner_id": cfg["owner_id"],
        "position_key": position_key,
        "ticker": str(ticker).upper().strip(),
        "position_state": str(position_state),
        "entry": float(entry),
        "initial_shares": float(initial_shares),
        "remaining_shares": float(remaining_shares),
        "realized_pl": float(realized_pl),
        "original_stop": float(original_stop),
        "current_stop_input": float(current_stop_input),
        "highest_r": (
            float(live_state.get("Highest R"))
            if live_state and pd.notna(live_state.get("Highest R"))
            else None
        ),
        "highest_state": live_state.get("Highest State") if live_state else None,
        "protective_floor": (
            float(live_state.get("Protective Stop Floor"))
            if live_state and pd.notna(live_state.get("Protective Stop Floor"))
            else None
        ),
        "last_live_mark": (
            float(live_state.get("Last Live Mark"))
            if live_state and pd.notna(live_state.get("Last Live Mark"))
            else None
        ),
        "last_action": live_state.get("Last Action") if live_state else None,
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    result = _phase4q5_request(
        "POST",
        "bullseye_positions",
        params={"on_conflict": "owner_id,position_key"},
        payload=payload,
        prefer="resolution=merge-duplicates,return=representation",
    )
    return {"ok": True, "status": "saved", "data": result}


def _phase4q5_load_latest_position(ticker):
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        return None

    rows = _phase4q5_request(
        "GET",
        "bullseye_positions",
        params={
            "select": "*",
            "owner_id": f'eq.{cfg["owner_id"]}',
            "ticker": f"eq.{str(ticker).upper().strip()}",
            "order": "updated_at.desc",
            "limit": "1",
        },
    )
    return rows[0] if rows else None


def _phase4q9_load_closed_trades(limit=100):
    """Return durable completed trades for the current Bullseye owner."""
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        return []
    rows = _phase4q5_request(
        "GET",
        "bullseye_closed_trades",
        params={
            "select": "*",
            "owner_id": f'eq.{cfg["owner_id"]}',
            "order": "closed_at.desc",
            "limit": str(int(limit)),
        },
    )
    return rows if isinstance(rows, list) else []


def _phase4q6_list_held_positions():
    """Return the owner's currently open durable swing positions, newest row per ticker."""
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        return []

    rows = _phase4q5_request(
        "GET",
        "bullseye_positions",
        params={
            "select": "ticker,entry,remaining_shares,realized_pl,highest_r,highest_state,protective_floor,last_live_mark,last_action,updated_at",
            "owner_id": f'eq.{cfg["owner_id"]}',
            "position_state": "eq.Entered / Live Position",
            "remaining_shares": "gt.0",
            "order": "updated_at.desc",
        },
    ) or []

    # Keep the most recently updated durable record for each ticker.
    held = []
    seen = set()
    for row in rows:
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        held.append(row)

    return sorted(held, key=lambda r: str(r.get("ticker", "")).upper())


def _phase4q6_load_held_ticker_callback(ticker):
    """One-click held-position selector: set ticker, then reuse the validated 4Q.5 loader."""
    st.session_state["phase4q1_ticker_key"] = str(ticker).upper().strip()
    _phase4q5_load_position_callback()


def _phase4q6_enrich_held_positions(rows):
    """Add live mark, P/L, R and attention state for the held-position dashboard."""
    enriched = []

    for row in rows:
        item = dict(row)
        ticker = str(item.get("ticker", "")).upper().strip()
        entry = float(item.get("entry") or 0.0)
        remaining = float(item.get("remaining_shares") or 0.0)
        original_stop = float(item.get("protective_floor") or 0.0)

        mark_info = get_position_mark(ticker)
        if mark_info.get("status") == "OK" and pd.notna(mark_info.get("price")):
            mark = float(mark_info["price"])
            mark_session = mark_info.get("session", "")
        else:
            mark = float(item.get("last_live_mark") or 0.0)
            mark_session = "Saved fallback"

        unrealized = (mark - entry) * remaining if entry > 0 and remaining > 0 and mark > 0 else np.nan
        risk_per_share = entry - original_stop if original_stop > 0 and original_stop < entry else np.nan
        current_r = (mark - entry) / risk_per_share if pd.notna(risk_per_share) and risk_per_share > 0 else np.nan

        highest_state = item.get("highest_state") or "Monitor"
        protective_floor = float(item.get("protective_floor") or 0.0)

        if protective_floor > 0 and mark > 0 and mark <= protective_floor:
            attention = "EXIT / REVIEW"
        elif pd.notna(current_r) and current_r < 0:
            attention = "Monitor"
        elif highest_state in ("Protect", "Trim", "Trail"):
            attention = highest_state
        else:
            attention = highest_state or "Hold"

        item.update({
            "mark": mark,
            "mark_session": mark_session,
            "unrealized_pl": unrealized,
            "current_r": current_r,
            "attention": attention,
        })
        enriched.append(item)

    return enriched




# -----------------------------------------------------------------------------
# Phase 4R.5 — Candidate Outcome Journal
# Measurement only: this layer records observed post-candidate events. It never
# changes Bullseye 4.0 scoring, 4R.3 Opportunity State, or trade-management math.
# -----------------------------------------------------------------------------
def _phase4r5_start_or_refresh_candidate(ticker, scored, trade_plan, current_mark):
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        return None
    ticker = str(ticker).upper().strip()
    mark = _phase4r2_num(current_mark)
    payload = {
        "owner_id": cfg["owner_id"], "ticker": ticker,
        "tracking_status": "TRACKING",
        "initial_stage": str(scored.get("4R Stage") or ""),
        "initial_opportunity_state": str(scored.get("4R.3 Opportunity State") or ""),
        "initial_score": _phase4r2_num(scored.get("Bullseye 4.0 Score")),
        "initial_price": mark,
        "entry_low": _phase4r2_num(trade_plan.get("Pullback Entry Low")),
        "entry_high": _phase4r2_num(trade_plan.get("Pullback Entry High")),
        "breakout_reference": _phase4r2_num(trade_plan.get("Breakout Reference")),
        "invalidation_reference": _phase4r2_num(trade_plan.get("Invalidation Reference")),
        "target_1r": _phase4r2_num(trade_plan.get("Target 1R")),
        "target_2r": _phase4r2_num(trade_plan.get("Target 2R")),
        "target_3r": _phase4r2_num(trade_plan.get("Target 3R")),
        "latest_price": mark, "max_observed_price": mark, "min_observed_price": mark,
        "latest_stage": str(scored.get("4R Stage") or ""),
        "latest_opportunity_state": str(scored.get("4R.3 Opportunity State") or ""),
        "latest_score": _phase4r2_num(scored.get("Bullseye 4.0 Score")),
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    return _phase4q5_request("POST", "bullseye_candidate_outcomes",
        params={"on_conflict":"owner_id,ticker"}, payload=payload,
        prefer="resolution=merge-duplicates,return=representation")


def _phase4r5_list_outcomes(ticker=None):
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]: return []
    params={"select":"*", "owner_id":f'eq.{cfg["owner_id"]}', "order":"tracked_at.desc"}
    if ticker: params["ticker"] = f"eq.{str(ticker).upper().strip()}"
    return _phase4q5_request("GET", "bullseye_candidate_outcomes", params=params) or []


def _phase4r5_update_from_scan(result_df):
    """Update only already-tracked candidates using observed scanner prices."""
    cfg = _phase4q5_storage_config()
    if not cfg["configured"] or result_df is None or result_df.empty: return 0
    tracked = {str(r.get("ticker","")).upper():r for r in _phase4r5_list_outcomes() if r.get("tracking_status")=="TRACKING"}
    now = pd.Timestamp.now(tz="UTC").isoformat(); updated=0
    for _, row in result_df.iterrows():
        t=str(row.get("Ticker","")).upper().strip()
        if t not in tracked: continue
        old=tracked[t]; p=_phase4r2_num(row.get("Price"))
        if p is None: continue
        def num(k): return _phase4r2_num(old.get(k))
        hi=max([x for x in [num("max_observed_price"),p] if x is not None])
        lo=min([x for x in [num("min_observed_price"),p] if x is not None])
        entry_lo,entry_hi=num("entry_low"),num("entry_high")
        breakout,inv=num("breakout_reference"),num("invalidation_reference")
        t1,t2,t3=num("target_1r"),num("target_2r"),num("target_3r")
        patch={"latest_price":p,"max_observed_price":hi,"min_observed_price":lo,
               "latest_stage":str(row.get("4R Stage") or ""),
               "latest_opportunity_state":str(row.get("4R.3 Opportunity State") or ""),
               "latest_score":_phase4r2_num(row.get("Bullseye 4.0 Score")),"updated_at":now}
        events=[
          ("entry_zone_seen_at", entry_lo is not None and entry_hi is not None and entry_lo <= p <= entry_hi),
          ("breakout_seen_at", breakout is not None and p >= breakout),
          ("target_1r_seen_at", t1 is not None and p >= t1),
          ("target_2r_seen_at", t2 is not None and p >= t2),
          ("target_3r_seen_at", t3 is not None and p >= t3),
          ("invalidation_seen_at", inv is not None and p <= inv),
          ("qualified_seen_at", _phase4r2_num(row.get("Bullseye 4.0 Score")) is not None and float(row.get("Bullseye 4.0 Score")) >= 90),
        ]
        for k,hit in events:
            if hit and not old.get(k): patch[k]=now
        _phase4q5_request("PATCH","bullseye_candidate_outcomes",
            params={"owner_id":f'eq.{cfg["owner_id"]}',"ticker":f"eq.{t}"}, payload=patch, prefer="return=minimal")
        updated += 1
    return updated


def _phase4r5_frame(records):
    if not records: return pd.DataFrame()
    df=pd.DataFrame(records)
    yes=lambda c: df[c].notna().map({True:"✅",False:"—"}) if c in df.columns else "—"
    out=pd.DataFrame({
      "Ticker":df.get("ticker"), "Status":df.get("tracking_status"),
      "Initial Stage":df.get("initial_stage"), "Initial Opportunity":df.get("initial_opportunity_state"),
      "Initial Score":df.get("initial_score"), "Latest Score":df.get("latest_score"),
      "Initial Price":df.get("initial_price"), "Latest Price":df.get("latest_price"),
      "Max Observed":df.get("max_observed_price"), "Min Observed":df.get("min_observed_price"),
      "Entry Zone":yes("entry_zone_seen_at"), "Qualified":yes("qualified_seen_at"),
      "Breakout":yes("breakout_seen_at"), "+1R":yes("target_1r_seen_at"), "+2R":yes("target_2r_seen_at"), "+3R":yes("target_3r_seen_at"),
      "Invalidation":yes("invalidation_seen_at"), "Tracked At":df.get("tracked_at"), "Updated At":df.get("updated_at")})
    return out

def _phase4q8_list_candidates():
    """Return the owner's saved candidate watchlist from Supabase."""
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        return []

    rows = _phase4q5_request(
        "GET",
        "bullseye_candidates",
        params={
            "select": "*",
            "owner_id": f'eq.{cfg["owner_id"]}',
            "order": "updated_at.desc",
        },
    ) or []
    return rows


def _phase4q8_save_candidate(ticker, scored, trade_plan, investigation, current_mark):
    """Upsert the current investigative snapshot into the durable candidate watchlist."""
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        return {"ok": False, "status": "not_configured"}

    ticker = str(ticker).upper().strip()
    payload = {
        "owner_id": cfg["owner_id"],
        "ticker": ticker,
        "candidate_action": str(investigation.get("Candidate Action") or ""),
        "action_reason": str(investigation.get("Action Reason") or ""),
        "bullseye_score": (
            float(scored.get("Bullseye 4.0 Score"))
            if pd.notna(scored.get("Bullseye 4.0 Score"))
            else None
        ),
        "signal_tier": str(scored.get("4H Signal Tier") or ""),
        "bullseye_action": str(scored.get("4I Action") or ""),
        "setup_quality": (
            float(scored.get("Setup Quality"))
            if pd.notna(scored.get("Setup Quality"))
            else None
        ),
        "rsi": float(scored.get("RSI")) if pd.notna(scored.get("RSI")) else None,
        "dist_20ma_pct": (
            float(scored.get("Dist 20MA %"))
            if pd.notna(scored.get("Dist 20MA %"))
            else None
        ),
        "rel_vol": (
            float(scored.get("Rel Vol"))
            if pd.notna(scored.get("Rel Vol"))
            else None
        ),
        "rs_vs_spy_20d": (
            float(scored.get("RS vs SPY 20D"))
            if pd.notna(scored.get("RS vs SPY 20D"))
            else None
        ),
        "market_regime": (
            float(scored.get("Market Regime"))
            if pd.notna(scored.get("Market Regime"))
            else None
        ),
        "current_mark": float(current_mark) if pd.notna(current_mark) else None,
        "entry_low": float(trade_plan.get("Pullback Entry Low")),
        "entry_high": float(trade_plan.get("Pullback Entry High")),
        "breakout_reference": float(trade_plan.get("Breakout Reference")),
        "invalidation_reference": float(trade_plan.get("Invalidation Reference")),
        "target_1r": float(trade_plan.get("Target 1R")),
        "target_2r": float(trade_plan.get("Target 2R")),
        "target_3r": float(trade_plan.get("Target 3R")),
        "risk_pct": float(trade_plan.get("Risk %")),
        "entry_mode": str(trade_plan.get("Entry Mode") or ""),
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    result = _phase4q5_request(
        "POST",
        "bullseye_candidates",
        params={"on_conflict": "owner_id,ticker"},
        payload=payload,
        prefer="resolution=merge-duplicates,return=representation",
    )
    return {"ok": True, "status": "saved", "data": result}


def _phase4q8_delete_candidate(ticker):
    """Remove one ticker from the saved candidate watchlist only."""
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        return False

    _phase4q5_request(
        "DELETE",
        "bullseye_candidates",
        params={
            "owner_id": f'eq.{cfg["owner_id"]}',
            "ticker": f"eq.{str(ticker).upper().strip()}",
        },
        prefer="return=minimal",
    )
    return True


def _phase4q8_load_candidate_callback():
    """Load the selected watchlist ticker into Candidate / Watching investigative mode."""
    ticker = str(st.session_state.get("phase4q8_selected_candidate", "")).upper().strip()
    if not ticker:
        return
    st.session_state["phase4q1_state_key"] = "Candidate / Watching"
    st.session_state["phase4q1_ticker_key"] = ticker
    st.session_state["phase4q1_view_active"] = True
    st.session_state["phase4q8_message"] = f"Loaded {ticker} into Candidate / Watching."


def _phase4q8_promote_candidate_callback():
    """Move a saved candidate into Live Position entry mode while preserving its reference snapshot."""
    ticker = str(st.session_state.get("phase4q8_selected_candidate", "")).upper().strip()
    st.session_state["phase4q8_message"] = ""
    if not ticker:
        st.session_state["phase4q8_message"] = "Select a saved candidate first."
        return

    snapshot = {}
    try:
        for row in _phase4q8_list_candidates():
            if str(row.get("ticker", "")).upper().strip() == ticker:
                snapshot = dict(row)
                break
    except Exception:
        snapshot = {}

    st.session_state["phase4q8_promotion_active"] = True
    st.session_state["phase4q8_promotion_ticker"] = ticker
    st.session_state["phase4q8_promotion_snapshot"] = snapshot

    st.session_state["phase4q1_state_key"] = "Entered / Live Position"
    st.session_state["phase4q1_ticker_key"] = ticker
    st.session_state["phase4q1_view_active"] = True

    # Promotion is not an execution record. Require the real trade data.
    st.session_state["phase4q1_actual_entry_key"] = 0.0
    st.session_state["phase4q1_initial_shares_key"] = 0.0
    st.session_state["phase4q1_remaining_shares_key"] = 0.0
    st.session_state["phase4q1_realized_pl_key"] = 0.0
    st.session_state["phase4q1_original_stop_key"] = 0.0
    st.session_state["phase4q1_current_stop_key"] = 0.0

    st.session_state["phase4q8_message"] = (
        f"Promoted {ticker} to Live Position entry mode. "
        "Enter the actual fill, shares and original stop, then Save / Update Position. "
        "The saved candidate remains in the watchlist until you explicitly delete it."
    )


def _phase4q8_delete_candidate_callback():
    """Delete only the currently selected watchlist candidate."""
    ticker = str(st.session_state.get("phase4q8_selected_candidate", "")).upper().strip()
    st.session_state["phase4q8_message"] = ""
    if not ticker:
        st.session_state["phase4q8_message"] = "Select a saved candidate first."
        return
    try:
        _phase4q8_delete_candidate(ticker)
        st.session_state["phase4q8_message"] = f"Removed {ticker} from Saved Candidates."
        st.session_state["phase4q8_selected_candidate"] = ""
    except Exception as exc:
        st.session_state["phase4q8_message"] = f"Delete failed: {exc}"


def _phase4q5_delete_position(ticker, entry):
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        return False

    _phase4q5_request(
        "DELETE",
        "bullseye_positions",
        params={
            "owner_id": f'eq.{cfg["owner_id"]}',
            "position_key": f"eq.{_phase4q5_position_key(ticker, entry)}",
        },
        prefer="return=minimal",
    )
    return True


def _phase4q9_close_trade(
    ticker,
    entry,
    final_exit_price,
    final_realized_pl,
    exit_reason="",
    notes="",
):
    """Atomically archive one legitimate live trade and remove its live-position row."""
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        return {"ok": False, "status": "not_configured"}

    endpoint = f'{cfg["url"]}/rest/v1/rpc/bullseye_close_trade'
    headers = {
        "apikey": cfg["key"],
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "p_owner_id": cfg["owner_id"],
        "p_position_key": _phase4q5_position_key(ticker, entry),
        "p_final_exit_price": float(final_exit_price),
        "p_final_realized_pl": float(final_realized_pl),
        "p_exit_reason": str(exit_reason or ""),
        "p_notes": str(notes or ""),
    }

    req = urllib_request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=12) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw) if raw else None
            return {"ok": True, "status": "closed", "data": data}
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Close-trade RPC HTTP {exc.code}: {detail[:700]}") from exc
    except Exception as exc:
        raise RuntimeError(f"Close-trade RPC failed: {exc}") from exc


def _phase4q5_seed_live_state_from_row(row):
    if not row:
        return

    ticker = str(row.get("ticker", "")).upper().strip()
    entry = float(row.get("entry") or 0.0)
    if not ticker or entry <= 0:
        return

    key = _phase4q4_state_key(ticker, entry)
    current = st.session_state["phase4q4_live_state"].get(key, {})

    durable = {
        "Ticker": ticker,
        "Entry": entry,
        "Highest R": row.get("highest_r", np.nan),
        "Highest State": row.get("highest_state") or "Monitor",
        "Highest State Rank": PHASE4Q4_STATE_RANK.get(row.get("highest_state") or "Monitor", 1),
        "Protective Stop Floor": row.get("protective_floor", np.nan),
        "Remaining Shares": float(row.get("remaining_shares") or 0.0),
        "Last Live Mark": row.get("last_live_mark", np.nan),
        "Last Action": row.get("last_action") or "",
        "Updated At": row.get("updated_at") or "",
    }

    # Merge so durable earned protection can never be weakened by a fresh session.
    if current:
        for field in ("Highest R", "Protective Stop Floor"):
            a = current.get(field, np.nan)
            b = durable.get(field, np.nan)
            if pd.notna(a) and pd.notna(b):
                durable[field] = max(float(a), float(b))
            elif pd.notna(a):
                durable[field] = a

        current_rank = int(current.get("Highest State Rank", -1))
        durable_rank = int(durable.get("Highest State Rank", -1))
        if current_rank > durable_rank:
            durable["Highest State"] = current.get("Highest State")
            durable["Highest State Rank"] = current_rank

    st.session_state["phase4q4_live_state"][key] = durable


def _phase4q5_load_position_callback():
    ticker = str(st.session_state.get("phase4q1_ticker_key", "")).upper().strip()
    st.session_state["phase4q5_last_message"] = ""
    if not ticker:
        st.session_state["phase4q5_last_message"] = "Enter a ticker before loading."
        return

    try:
        row = _phase4q5_load_latest_position(ticker)
        if not row:
            st.session_state["phase4q5_last_message"] = f"No durable saved position found for {ticker}."
            return

        st.session_state["phase4q1_state_key"] = row.get("position_state") or "Entered / Live Position"
        st.session_state["phase4q1_entry_key"] = float(row.get("entry") or 0.0)
        st.session_state["phase4q1_initial_shares_key"] = float(row.get("initial_shares") or 0.0)
        st.session_state["phase4q1_remaining_shares_key"] = float(row.get("remaining_shares") or 0.0)
        st.session_state["phase4q1_realized_pl_key"] = float(row.get("realized_pl") or 0.0)
        st.session_state["phase4q1_initial_stop_key"] = float(row.get("original_stop") or 0.0)
        st.session_state["phase4q1_actual_stop_key"] = float(row.get("current_stop_input") or 0.0)
        st.session_state["phase4q1_view_active"] = True
        _phase4q5_seed_live_state_from_row(row)
        st.session_state["phase4q5_last_loaded"] = row
        st.session_state["phase4q5_last_message"] = f"Loaded durable {ticker} position."
    except Exception as exc:
        st.session_state["phase4q5_last_message"] = f"Load failed: {exc}"


PHASE4Q4_STATE_RANK = {
    "Exit": 0,
    "Monitor": 1,
    "Hold": 2,
    "Protect": 3,
    "Trim": 4,
    "Trail": 5,
}

def _phase4q4_state_key(ticker, entry):
    return f"{str(ticker).upper().strip()}|{float(entry):.4f}"

def _phase4q4_get_state(ticker, entry):
    return st.session_state["phase4q4_live_state"].get(_phase4q4_state_key(ticker, entry))

def _phase4q4_store_state(ticker, entry, payload):
    st.session_state["phase4q4_live_state"][_phase4q4_state_key(ticker, entry)] = payload

def _phase4q4_clear_state(ticker, entry):
    st.session_state["phase4q4_live_state"].pop(_phase4q4_state_key(ticker, entry), None)

def _phase4q4_merge_live_state(ticker, entry, current_r, management_state, management_action,
                               protective_stop, remaining_shares, mark):
    """Persist live progress within the Streamlit session; never ratchet backward."""
    prior = _phase4q4_get_state(ticker, entry)
    now = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M:%S %Z")

    if prior is None:
        merged = {
            "Ticker": str(ticker).upper().strip(),
            "Entry": float(entry),
            "Highest R": float(current_r) if pd.notna(current_r) else np.nan,
            "Highest State": management_state,
            "Highest State Rank": PHASE4Q4_STATE_RANK.get(management_state, -1),
            "Protective Stop Floor": float(protective_stop) if pd.notna(protective_stop) else np.nan,
            "Remaining Shares": float(remaining_shares),
            "Last Live Mark": float(mark),
            "Last Action": management_action,
            "Updated At": now,
        }
    else:
        merged = dict(prior)

        prior_r = prior.get("Highest R", np.nan)
        if pd.notna(current_r):
            merged["Highest R"] = float(current_r) if pd.isna(prior_r) else max(float(prior_r), float(current_r))

        prior_rank = int(prior.get("Highest State Rank", -1))
        current_rank = PHASE4Q4_STATE_RANK.get(management_state, -1)
        if current_rank >= prior_rank:
            merged["Highest State"] = management_state
            merged["Highest State Rank"] = current_rank

        prior_floor = prior.get("Protective Stop Floor", np.nan)
        if pd.notna(protective_stop):
            merged["Protective Stop Floor"] = (
                float(protective_stop)
                if pd.isna(prior_floor)
                else max(float(prior_floor), float(protective_stop))
            )

        merged["Remaining Shares"] = float(remaining_shares)
        merged["Last Live Mark"] = float(mark)
        merged["Last Action"] = management_action
        merged["Updated At"] = now

    _phase4q4_store_state(ticker, entry, merged)
    return merged



def _phase4q4_reset_test_state():
    st.session_state["phase4q4_test_state"] = None

def _phase4q4_apply_test_step(test_r, test_state, test_floor):
    """Isolated ratchet test; never writes to real live-position state."""
    prior = st.session_state.get("phase4q4_test_state")
    if prior is None:
        prior = {
            "Highest R": -999.0,
            "Highest State": "Monitor",
            "Highest State Rank": PHASE4Q4_STATE_RANK["Monitor"],
            "Protective Stop Floor": float("-inf"),
        }

    merged = dict(prior)
    merged["Highest R"] = max(float(prior.get("Highest R", -999.0)), float(test_r))

    current_rank = PHASE4Q4_STATE_RANK.get(test_state, -1)
    prior_rank = int(prior.get("Highest State Rank", -1))
    if current_rank >= prior_rank:
        merged["Highest State"] = test_state
        merged["Highest State Rank"] = current_rank

    merged["Protective Stop Floor"] = max(
        float(prior.get("Protective Stop Floor", float("-inf"))),
        float(test_floor),
    )

    st.session_state["phase4q4_test_state"] = merged
    return merged


def build_live_position_management(
    entry,
    mark,
    original_stop,
    raw_current_stop_input,
    active_stop,
    remaining_shares,
    bullseye_invalidation,
):
    """Phase 4Q.2 management overlay; does not alter Bullseye scoring."""
    out = {"state":"Monitor","action":"Hold / Monitor","reason":"Insufficient live-position data.",
           "current_r":np.nan,"current_r_raw":np.nan,"t1":np.nan,"t2":np.nan,"t3":np.nan,
           "protective_stop":np.nan,"protective_stop_source":"Unavailable",
           "raw_current_stop_input":raw_current_stop_input,
           "resolved_active_stop":active_stop}
    if entry <= 0 or mark <= 0 or remaining_shares <= 0:
        return out
    stop_basis = original_stop if original_stop > 0 else bullseye_invalidation
    if stop_basis <= 0 or stop_basis >= entry:
        out["reason"] = "A valid original stop below entry is required for R-based management."
        return out
    risk = entry - stop_basis
    r_raw = (mark - entry) / risk
    r = round(r_raw, 4)
    t1, t2, t3 = entry + risk, entry + 2 * risk, entry + 3 * risk
    if raw_current_stop_input > 0:
        live_stop = raw_current_stop_input
        stop_source = "User-entered current stop"
    elif original_stop > 0:
        live_stop = original_stop
        stop_source = "Original stop at entry"
    else:
        live_stop = bullseye_invalidation
        stop_source = "Bullseye current invalidation fallback"
    out.update(
        current_r=r,
        current_r_raw=r_raw,
        t1=t1,
        t2=t2,
        t3=t3,
        protective_stop=live_stop,
        protective_stop_source=stop_source,
    )
    if live_stop > 0 and mark <= live_stop:
        out.update(state="Exit",action="Exit / Review Immediately",
                   reason="Position Mark is at or below the active stop / current invalidation.")
    elif r < 0:
        out.update(
            state="Monitor",
            action="Hold / Monitor",
            reason=f"Position is below entry but remains above the ${live_stop:,.2f} protective stop reference. Continue monitoring.",
        )
    elif r < 1:
        out.update(state="Hold",action="Hold",
                   reason="Position is profitable but has not yet reached +1R.")
    elif r < 2:
        out.update(state="Protect",action="Protect / Consider Breakeven",
                   reason="Position has reached +1R; begin protecting original risk.",
                   protective_stop=max(entry,live_stop),
                   protective_stop_source=stop_source)
    elif r < 3:
        out.update(state="Trim",action="Trim / Trail",
                   reason="Position has reached +2R; consider partial profit and trail the remainder.",
                   protective_stop=max(t1,live_stop),
                   protective_stop_source=stop_source)
    else:
        out.update(state="Trail",action="Trail / Protect Winner",
                   reason="Position is at +3R or better; prioritize protecting accumulated gains.",
                   protective_stop=max(t2,live_stop),
                   protective_stop_source=stop_source)
    return out


def build_phase4r4_profit_protection(entry, mark, t1, t2, t3, current_r, highest_r=np.nan):
    """Phase 4R.4 observational target-approach layer; never changes 4Q.2 management."""
    result = {
        "state": "BUILDING",
        "icon": "🏃",
        "signal": "🏃 BUILDING",
        "distance_to_t1_pct": np.nan,
        "progress_to_t1_pct": np.nan,
        "peak_progress_pct": np.nan,
        "why": "Position is progressing toward the first +1R profit-protection reference.",
        "next_trigger": "Continue monitoring progress toward +1R.",
    }
    vals = [entry, mark, t1]
    if any(pd.isna(x) for x in vals) or float(entry) <= 0 or float(t1) <= float(entry):
        result.update(
            state="NOT AVAILABLE", icon="⏳", signal="⏳ NOT AVAILABLE",
            why="A valid actual entry and original-risk +1R target are required.",
            next_trigger="Enter or restore the original stop so Bullseye can calculate +1R progress.",
        )
        return result

    entry=float(entry); mark=float(mark); t1=float(t1)
    current_r_val = float(current_r) if pd.notna(current_r) else (mark-entry)/(t1-entry)
    highest_r_val = float(highest_r) if pd.notna(highest_r) else current_r_val
    highest_r_val = max(highest_r_val, current_r_val)
    progress = current_r_val * 100.0
    peak_progress = highest_r_val * 100.0
    distance_pct = (t1 / mark - 1.0) * 100.0 if mark > 0 else np.nan
    pullback_from_peak_r = highest_r_val - current_r_val

    result.update(
        distance_to_t1_pct=distance_pct,
        progress_to_t1_pct=progress,
        peak_progress_pct=peak_progress,
    )

    # Read-only observational states. Thresholds are deliberately simple so forward
    # evidence can validate them before any management rule is changed.
    if current_r_val >= 2.0:
        state, icon = "WINNER", "🏆"
        why = f"Position is at {current_r_val:.2f}R, beyond +1R and at/above the +2R winner-management region."
        next_trigger = f"Use existing 4Q.2 management; next reference is +3R at ${float(t3):,.2f}." if pd.notna(t3) else "Use existing 4Q.2 management to protect the winner."
    elif current_r_val >= 1.0:
        state, icon = "T1 REACHED", "🎯"
        why = f"The position has reached the first +1R objective and is currently at {current_r_val:.2f}R."
        next_trigger = f"Use existing 4Q.2 protection; +2R reference is ${float(t2):,.2f}." if pd.notna(t2) else "Use existing 4Q.2 protection for the remaining position."
    elif highest_r_val >= 0.75 and pullback_from_peak_r >= 0.15:
        state, icon = "PROTECT", "🛡️"
        why = f"The trade reached {peak_progress:.0f}% of +1R, then pulled back {pullback_from_peak_r:.2f}R; accumulated progress deserves attention."
        next_trigger = f"Watch for renewed strength toward +1R at ${t1:,.2f} or further deterioration; 4Q.2 remains authoritative."
    elif current_r_val >= 0.75:
        state, icon = "APPROACHING T1", "👀"
        why = f"The position has completed {progress:.0f}% of the move from actual entry to +1R."
        next_trigger = f"+1R is ${t1:,.2f}, {abs(distance_pct):.2f}% above the current mark." if pd.notna(distance_pct) else f"Watch +1R at ${t1:,.2f}."
    else:
        state, icon = "BUILDING", "🏃"
        why = f"The position has completed {progress:.0f}% of the move from actual entry to +1R."
        next_trigger = f"Approaching-T1 observation begins at 0.75R; +1R is ${t1:,.2f}."

    result.update(state=state, icon=icon, signal=f"{icon} {state}", why=why, next_trigger=next_trigger)
    return result


def one_symbol(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return None
        df = data[ticker].copy()
    else:
        df = data.copy()
    df = df.dropna(subset=["Close", "Volume"])
    return df if len(df) >= 220 else None

def pct(a, b):
    if b == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return (a / b - 1) * 100

def clamp(x, lo=0, hi=100):
    return float(np.clip(x, lo, hi))

def score_stock(df, spy):
    c = df["Close"]
    v = df["Volume"]
    last = float(c.iloc[-1])

    ma20 = c.rolling(20).mean().iloc[-1]
    ma50 = c.rolling(50).mean().iloc[-1]
    ma200 = c.rolling(200).mean().iloc[-1]

    r5 = pct(last, c.iloc[-6])
    r10 = pct(last, c.iloc[-11])
    r20 = pct(last, c.iloc[-21])
    r60 = pct(last, c.iloc[-61])

    # 1) Momentum — 20
    momentum = (
        np.clip((r5 + 2) * 1.0, 0, 5)
        + np.clip((r10 + 3) * 0.65, 0, 5)
        + np.clip((r20 + 5) * 0.45, 0, 5)
        + np.clip((r60 + 8) * 0.20, 0, 3)
        + (2 if last > ma20 > ma50 else 1 if last > ma20 else 0)
    )
    momentum = clamp(momentum, 0, 20)

    # 2) Volume — 15
    # Phase 2E calibration: score participation on a smoother curve so
    # normal/healthy volume does not collapse to zero.
    avg20v = v.tail(20).mean()
    avg5v = v.tail(5).mean()
    rv20 = float(v.iloc[-1] / avg20v) if avg20v else 0
    rv5 = float(avg5v / avg20v) if avg20v else 0

    # Today's relative volume: up to 8 points.
    if rv20 >= 1.50:
        today_vol_score = 8.0
    elif rv20 >= 1.20:
        today_vol_score = 6.5
    elif rv20 >= 1.00:
        today_vol_score = 5.0
    elif rv20 >= 0.80:
        today_vol_score = 3.5
    elif rv20 >= 0.60:
        today_vol_score = 2.0
    else:
        today_vol_score = 0.5

    # Five-day participation: up to 7 points.
    if rv5 >= 1.30:
        recent_vol_score = 7.0
    elif rv5 >= 1.10:
        recent_vol_score = 5.5
    elif rv5 >= 0.95:
        recent_vol_score = 4.0
    elif rv5 >= 0.80:
        recent_vol_score = 2.5
    elif rv5 >= 0.65:
        recent_vol_score = 1.5
    else:
        recent_vol_score = 0.5

    volume = clamp(today_vol_score + recent_vol_score, 0, 15)

    # 3) Relative strength vs SPY — 15
    spyc = spy["Close"]
    stock5 = pct(last, c.iloc[-6])
    stock20 = pct(last, c.iloc[-21])
    stock60 = pct(last, c.iloc[-61])
    spy5 = pct(spyc.iloc[-1], spyc.iloc[-6])
    spy20_perf = pct(spyc.iloc[-1], spyc.iloc[-21])
    spy60 = pct(spyc.iloc[-1], spyc.iloc[-61])

    rs5 = stock5 - spy5
    rs20 = stock20 - spy20_perf
    rs60 = stock60 - spy60
    relative_strength = clamp(
        np.clip((rs5 + 1) * 1.4, 0, 5)
        + np.clip((rs20 + 2) * 0.8, 0, 5)
        + np.clip((rs60 + 3) * 0.45, 0, 5),
        0, 15
    )

    # 4) Technical setup — 20
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs14 = gain / loss.replace(0, np.nan)
    rsi = float(100 - 100 / (1 + rs14.iloc[-1])) if pd.notna(rs14.iloc[-1]) else 50

    macd_fast = c.ewm(span=12, adjust=False).mean()
    macd_slow = c.ewm(span=26, adjust=False).mean()
    macd = macd_fast - macd_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_bull = macd.iloc[-1] > signal.iloc[-1]

    high20_prev = c.iloc[-21:-1].max()
    near_breakout = last >= high20_prev * 0.98
    breakout = last >= high20_prev

    trend_points = (
        4 if last > ma20 > ma50 > ma200
        else 3 if last > ma20 > ma50
        else 1 if last > ma20
        else 0
    )
    technical = (
        trend_points
        + (5 if breakout else 3 if near_breakout else 0)
        + (4 if macd_bull else 0)
        + (4 if 50 <= rsi <= 72 else 2 if 45 <= rsi < 50 else 0)
        + (3 if last > ma200 else 0)
    )
    technical = clamp(technical, 0, 20)

    # 5) Setup quality / entry timing — 15
    dist20 = pct(last, ma20)
    dist50 = pct(last, ma50)
    daily_ret = c.pct_change()
    vol_recent = float(daily_ret.tail(10).std())
    vol_prior = float(daily_ret.iloc[-30:-10].std())
    vol_expansion = (vol_recent / vol_prior) if vol_prior and not pd.isna(vol_prior) else 1.0
    prior5 = pct(c.iloc[-6], c.iloc[-11])
    accel = r5 - prior5

    setup = 0.0
    setup += 5 if 0 <= dist20 <= 5 else 3 if (-2 <= dist20 < 0 or 5 < dist20 <= 8) else 1 if 8 < dist20 <= 12 else 0
    setup += 3 if 0 <= dist50 <= 10 else 1.5 if (-3 <= dist50 < 0 or 10 < dist50 <= 15) else 0
    setup += 3 if breakout and rv20 >= 1.0 else 2 if near_breakout else 0
    setup += 2 if 0 < accel <= 6 else 1 if accel > 6 else 0
    setup += 2 if 1.0 <= vol_expansion <= 1.6 else 1 if 0.8 <= vol_expansion < 1.0 else 0
    if dist20 > 12:
        setup -= 4
    elif dist20 > 8:
        setup -= 2
    if rsi > 78:
        setup -= 3
    elif rsi > 72:
        setup -= 1
    setup = clamp(setup, 0, 15)

    # 6) Market regime — 10
    spy20 = spyc.rolling(20).mean().iloc[-1]
    spy50 = spyc.rolling(50).mean().iloc[-1]
    spy200 = spyc.rolling(200).mean().iloc[-1]
    market_regime = (
        10 if spyc.iloc[-1] > spy20 > spy50 > spy200
        else 7 if spyc.iloc[-1] > spy50 > spy200
        else 4 if spyc.iloc[-1] > spy200
        else 1
    )

    # 7) Risk/liquidity — 5
    dollar_vol = float((c * v).tail(20).mean())
    annualized_vol = float(c.pct_change().tail(20).std() * math.sqrt(252) * 100)
    risk = 0
    risk += 2 if dollar_vol >= 100_000_000 else 1.5 if dollar_vol >= 50_000_000 else 0.5 if dollar_vol >= 10_000_000 else 0
    risk += 2 if annualized_vol < 35 else 1.5 if annualized_vol < 50 else 0.5 if annualized_vol < 70 else 0
    risk += 0.5 if last > 5 else 0
    risk += 0.5 if last > ma50 else 0
    risk = clamp(risk, 0, 5)

    raw_total = momentum + volume + relative_strength + technical + setup + market_regime + risk

    # Phase 2C: overextension / anti-chase controls
    extension_penalty = 0.0

    if accel > 30:
        extension_penalty += 12
    elif accel > 20:
        extension_penalty += 8
    elif accel > 12:
        extension_penalty += 4
    elif accel > 8:
        extension_penalty += 2

    if dist20 > 20:
        extension_penalty += 10
    elif dist20 > 15:
        extension_penalty += 7
    elif dist20 > 10:
        extension_penalty += 4
    elif dist20 > 7:
        extension_penalty += 2

    if rsi > 82:
        extension_penalty += 6
    elif rsi > 77:
        extension_penalty += 4
    elif rsi > 72:
        extension_penalty += 2

    adjusted_total = raw_total - extension_penalty

    if setup <= 2:
        adjusted_total = min(adjusted_total, 72)
    elif setup <= 5:
        adjusted_total = min(adjusted_total, 79)
    elif setup <= 8:
        adjusted_total = min(adjusted_total, 86)

    total = round(clamp(adjusted_total, 0, 100), 1)

    if setup <= 2 and (accel > 12 or dist20 > 10 or rsi > 77):
        label = "Don't Chase"
    elif total >= 85:
        label = "Exceptional"
    elif total >= 75:
        label = "Strong"
    elif total >= 65:
        label = "Bullish"
    elif total >= 50:
        label = "Watch"
    else:
        label = "Avoid"

    # Phase 2D: dedicated 1–4 week Opportunity Score
    opportunity_score = (
        (setup / 15) * 30
        + (relative_strength / 15) * 20
        + (volume / 15) * 15
        + (momentum / 20) * 15
        + (technical / 20) * 10
        + (market_regime / 10) * 10
        - (extension_penalty * 1.5)
    )
    opportunity_score = round(clamp(opportunity_score, 0, 100), 1)

    if opportunity_score >= 85:
        opportunity_label = "Prime Setup"
    elif opportunity_score >= 75:
        opportunity_label = "Attractive"
    elif opportunity_score >= 65:
        opportunity_label = "Promising"
    elif opportunity_score >= 50:
        opportunity_label = "Watch"
    else:
        opportunity_label = "Low Priority"

    # Phase 3C: Experimental Bullseye 3.0 Score
    # Built from signals that held up best in multi-period validation.

    exp_rs = clamp((relative_strength / 15) * 30, 0, 30)

    if dist20 >= 20:
        exp_dist = 25
    elif dist20 >= 15:
        exp_dist = 22
    elif dist20 >= 10:
        exp_dist = 19
    elif dist20 >= 5:
        exp_dist = 14
    elif dist20 >= 0:
        exp_dist = 9
    elif dist20 >= -5:
        exp_dist = 6
    else:
        exp_dist = 4

    if 70 <= rsi < 80:
        exp_rsi = 15
    elif rsi >= 80:
        exp_rsi = 14
    elif 60 <= rsi < 70:
        exp_rsi = 11
    elif 50 <= rsi < 60:
        exp_rsi = 8
    elif 40 <= rsi < 50:
        exp_rsi = 6
    else:
        exp_rsi = 4

    exp_momentum = clamp((momentum / 20) * 12, 0, 12)
    exp_volume = clamp((volume / 15) * 8, 0, 8)
    exp_technical = clamp((technical / 20) * 5, 0, 5)
    exp_extension = clamp((extension_penalty / 20) * 5, 0, 5)

    experimental_score = round(
        clamp(
            exp_rs + exp_dist + exp_rsi + exp_momentum
            + exp_volume + exp_technical + exp_extension,
            0,
            100,
        ),
        1,
    )

    if experimental_score >= 80:
        experimental_label = "3.0 Strong"
    elif experimental_score >= 70:
        experimental_label = "3.0 Bullish"
    elif experimental_score >= 60:
        experimental_label = "3.0 Watch"
    else:
        experimental_label = "3.0 Low"

    # Phase 4A: Bullseye 4.0 prototype
    # Experimental 3.0 remains the base. Beta only adds weight when:
    #   1) the 3.0 setup is already strong, and
    #   2) the broader market regime is supportive.
    stock_ret_120 = c.pct_change().tail(120).dropna()
    spy_ret_120 = spyc.pct_change().tail(120).dropna()
    common_beta_dates = stock_ret_120.index.intersection(spy_ret_120.index)

    beta_120 = np.nan
    if len(common_beta_dates) >= 60:
        sr = stock_ret_120.loc[common_beta_dates]
        pr = spy_ret_120.loc[common_beta_dates]
        spy_var = float(pr.var())
        if spy_var > 0:
            beta_120 = float(sr.cov(pr) / spy_var)

    beta_strength = 0.0
    if pd.notna(beta_120):
        beta_strength = clamp((beta_120 - 1.0) / 1.0, 0, 1)

    setup_strength_4 = clamp((experimental_score - 60) / 20, 0, 1)
    market_support_4 = 1.0 if market_regime >= 7 else 0.0

    accelerator_4 = round(
        12.0 * beta_strength * setup_strength_4 * market_support_4,
        1,
    )

    bullseye4_score = round(
        clamp(experimental_score + accelerator_4, 0, 100),
        1,
    )

    # Phase 4B tuning variants.
    # Variant B1: no light accelerator. Require a meaningful beta/setup combination.
    accelerator_4b1 = accelerator_4 if accelerator_4 >= 4 else 0.0
    bullseye4b1_score = round(
        clamp(experimental_score + accelerator_4b1, 0, 100),
        1,
    )

    # Variant B2: stricter threshold. Require stronger setup, supportive market,
    # and beta above ~1.5 before any boost is allowed.
    accelerator_4b2 = 0.0
    if (
        pd.notna(beta_120)
        and experimental_score >= 70
        and market_regime >= 7
        and beta_120 >= 1.5
    ):
        beta_factor_b2 = clamp((beta_120 - 1.5) / 0.75, 0, 1)
        setup_factor_b2 = clamp((experimental_score - 70) / 15, 0, 1)
        accelerator_4b2 = round(12.0 * beta_factor_b2 * setup_factor_b2, 1)

    bullseye4b2_score = round(
        clamp(experimental_score + accelerator_4b2, 0, 100),
        1,
    )

    # Variant B3: moderate-only accelerator. Only keep boosts >= 8 points.
    accelerator_4b3 = accelerator_4 if accelerator_4 >= 8 else 0.0
    bullseye4b3_score = round(
        clamp(experimental_score + accelerator_4b3, 0, 100),
        1,
    )

    if accelerator_4 >= 8:
        accelerator_label = "Strong accelerator"
    elif accelerator_4 >= 4:
        accelerator_label = "Moderate accelerator"
    elif accelerator_4 > 0:
        accelerator_label = "Light accelerator"
    else:
        accelerator_label = "No accelerator"

    if bullseye4_score >= 90:
        bullseye4_rating = "4.0 Prime"
    elif bullseye4_score >= 80:
        bullseye4_rating = "4.0 Strong"
    elif bullseye4_score >= 70:
        bullseye4_rating = "4.0 Bullish"
    elif bullseye4_score >= 60:
        bullseye4_rating = "4.0 Watch"
    else:
        bullseye4_rating = "4.0 Low"

    # Phase 4H: Final Signal Architecture
    avg_dollar_vol_60 = float((c * v).tail(60).mean()) if len(c) >= 60 else np.nan
    r120_live = pct(last, c.iloc[-121]) if len(c) >= 121 else np.nan

    core_count_4h = 0
    core_count_4h += 1 if pd.notna(avg_dollar_vol_60) and avg_dollar_vol_60 >= 5_000_000_000 else 0
    core_count_4h += 1 if pd.notna(r120_live) and r120_live >= 70 else 0
    core_count_4h += 1 if accelerator_4 >= 10 else 0
    core_count_4h += 1 if pd.notna(beta_120) and beta_120 >= 1.5 else 0

    if bullseye4_score >= 95 and core_count_4h >= 3:
        signal_tier_4h = "Elite Confirmed"
        signal_rank_4h = 5
    elif bullseye4_score >= 95 and accelerator_4 >= 10:
        signal_tier_4h = "Confirmed Prime"
        signal_rank_4h = 4
    elif bullseye4_score >= 95:
        signal_tier_4h = "Prime 95+"
        signal_rank_4h = 3
    elif bullseye4_score >= 92.5:
        signal_tier_4h = "Very High Conviction"
        signal_rank_4h = 2
    elif bullseye4_score >= 90:
        signal_tier_4h = "High Conviction"
        signal_rank_4h = 1
    else:
        signal_tier_4h = "Standard"
        signal_rank_4h = 0

    badges_4h = []
    if accelerator_4 >= 10:
        badges_4h.append("Accel>=10")
    if core_count_4h >= 3:
        badges_4h.append("Core>=3")
    if pd.notna(avg_dollar_vol_60) and avg_dollar_vol_60 >= 5_000_000_000:
        badges_4h.append("$5B+ Liquidity")
    signal_badges_4h = " | ".join(badges_4h) if badges_4h else "—"

    # Phase 4I: live decision labels.
    if signal_tier_4h == "Elite Confirmed":
        action_4i, action_rank_4i = "Priority Watch", 5
    elif signal_tier_4h == "Confirmed Prime":
        action_4i, action_rank_4i = "Strong Watch", 4
    elif signal_tier_4h == "Prime 95+":
        action_4i, action_rank_4i = "Watch Closely", 3
    elif signal_tier_4h == "Very High Conviction":
        action_4i, action_rank_4i = "Watch", 2
    elif signal_tier_4h == "High Conviction":
        action_4i, action_rank_4i = "Secondary Watch", 1
    else:
        action_4i, action_rank_4i = "Background", 0

    if signal_tier_4h == "Elite Confirmed":
        why_4i = "95+ score with 3+ core confirmations"
    elif signal_tier_4h == "Confirmed Prime":
        why_4i = "95+ score with Accelerator >=10"
    elif signal_tier_4h == "Prime 95+":
        why_4i = "95+ Bullseye 4.0 score"
    elif signal_tier_4h == "Very High Conviction":
        why_4i = "92.5+ Bullseye 4.0 score"
    elif signal_tier_4h == "High Conviction":
        why_4i = "90+ Bullseye 4.0 score"
    else:
        why_4i = "Below 90 high-conviction threshold"

    return {
        "Ticker": None,
        "Score": total,
        "Opportunity Score": opportunity_score,
        "Opportunity Rating": opportunity_label,
        "Experimental 3.0 Score": experimental_score,
        "Experimental 3.0 Rating": experimental_label,
        "Bullseye 4.0 Score": bullseye4_score,
        "Bullseye 4.0 Rating": bullseye4_rating,
        "4.0 Accelerator": accelerator_4,
        "4.0 Accelerator Label": accelerator_label,
        "Beta 120D": round(beta_120, 2) if pd.notna(beta_120) else np.nan,
        "4H Signal Tier": signal_tier_4h,
        "4H Signal Rank": signal_rank_4h,
        "4H Core Count": core_count_4h,
        "4H Signal Badges": signal_badges_4h,
        "4I Action": action_4i,
        "4I Action Rank": action_rank_4i,
        "4I Why": why_4i,
        "Avg $ Volume 60D ($M)": round(avg_dollar_vol_60 / 1_000_000, 1) if pd.notna(avg_dollar_vol_60) else np.nan,
        "120D %": round(float(r120_live), 2) if pd.notna(r120_live) else np.nan,
        "Bullseye 4B1 Score": bullseye4b1_score,
        "Bullseye 4B2 Score": bullseye4b2_score,
        "Bullseye 4B3 Score": bullseye4b3_score,
        "4B1 Accelerator": accelerator_4b1,
        "4B2 Accelerator": accelerator_4b2,
        "4B3 Accelerator": accelerator_4b3,
        "Rating": label,
        "Price": round(last, 2),
        "5D %": round(float(r5), 2),
        "20D %": round(float(r20), 2),
        "60D %": round(float(r60), 2),
        "Rel Vol": round(rv20, 2),
        "RS vs SPY 20D": round(float(rs20), 2),
        "RSI": round(rsi, 1),
        "Momentum": round(momentum, 1),
        "Volume": round(volume, 1),
        "Relative Strength": round(relative_strength, 1),
        "Technical": round(technical, 1),
        "Setup Quality": round(setup, 1),
        "Extension Penalty": round(extension_penalty, 1),
        "Dist 20MA %": round(float(dist20), 2),
        "Momentum Accel": round(float(accel), 2),
        "Market Regime": round(float(market_regime), 1),
        "Risk/Liquidity": round(risk, 1),
    }


def build_phase4r_early_warning(scored):
    """
    Phase 4R.1 — Developing Setup / Early-Warning layer.

    IMPORTANT:
    - Does NOT alter Bullseye 4.0 score, Phase 4H signal tiers, or Phase 4I actions.
    - Uses existing validated measurements only.
    - Purpose is to surface near-qualification setups earlier without lowering
      Bullseye's 90-point High Conviction threshold.
    """
    score = float(scored.get("Bullseye 4.0 Score", np.nan))
    accel = float(scored.get("4.0 Accelerator", np.nan))
    core = int(scored.get("4H Core Count", 0) or 0)
    mom_accel = float(scored.get("Momentum Accel", np.nan))
    rel_vol = float(scored.get("Rel Vol", np.nan))
    rs20 = float(scored.get("RS vs SPY 20D", np.nan))
    rsi = float(scored.get("RSI", np.nan))
    dist20 = float(scored.get("Dist 20MA %", np.nan))
    action_rank = int(scored.get("4I Action Rank", 0) or 0)

    if not np.isfinite(score):
        return {
            "4R Stage": "Unavailable",
            "4R Stage Rank": -1,
            "4R Readiness": 0,
            "4R Gap to 90": np.nan,
            "4R Why": "No valid Bullseye 4.0 score.",
            "4R Next Trigger": "Unavailable",
        }

    gap = max(0.0, 90.0 - score)

    # Supporting evidence already present in Bullseye.
    confirmations = []
    if np.isfinite(accel) and accel >= 8:
        confirmations.append("Accelerator>=8")
    if core >= 2:
        confirmations.append("Core>=2")
    if np.isfinite(mom_accel) and mom_accel > 0:
        confirmations.append("Momentum accelerating")
    if np.isfinite(rel_vol) and rel_vol >= 0.90:
        confirmations.append("RelVol>=0.90")
    if np.isfinite(rs20) and rs20 > 0:
        confirmations.append("RS>SPY")

    readiness = len(confirmations)

    # Qualification remains exactly where Phase 4H/4I already put it: 90+.
    if action_rank >= 1 or score >= 90:
        stage = "QUALIFIED"
        rank = 3
        why = f"Bullseye 4.0 score {score:.1f} is at/above the existing 90-point High Conviction threshold."
        next_trigger = "Use existing 4H/4I signal tier and Candidate Investigation."

    # Developing is intentionally conservative: reasonably close to 90 plus
    # multiple independent confirmations. This prevents an 'everything is almost ready' list.
    elif score >= 85 and readiness >= 2:
        stage = "DEVELOPING"
        rank = 2
        why = (
            f"{score:.1f} score is {gap:.1f} points below qualification with "
            f"{readiness} supporting confirmations: {', '.join(confirmations)}."
        )
        next_trigger = f"Needs +{gap:.1f} Bullseye points to reach 90."

    # Watch catches the broader near-threshold neighborhood but keeps it distinct
    # from the stronger Developing classification.
    elif score >= 80:
        stage = "WATCH"
        rank = 1
        why = (
            f"{score:.1f} score is below qualification. "
            f"{readiness} supporting confirmation{'s' if readiness != 1 else ''} currently present."
        )
        next_trigger = (
            f"Needs +{gap:.1f} points; watch for stronger momentum/volume/RS confirmation."
        )

    else:
        stage = "BACKGROUND"
        rank = 0
        why = f"{score:.1f} score is not yet in the Phase 4R near-qualification zone."
        next_trigger = f"Needs +{gap:.1f} points to reach the 90 qualification threshold."

    # Timing caution is informational only and never changes stage.
    cautions = []
    if np.isfinite(rsi) and rsi >= 78:
        cautions.append("RSI extended")
    if np.isfinite(dist20) and dist20 >= 12:
        cautions.append("far above 20MA")
    if cautions:
        why += " Timing caution: " + ", ".join(cautions) + "."

    return {
        "4R Stage": stage,
        "4R Stage Rank": rank,
        "4R Readiness": readiness,
        "4R Gap to 90": round(gap, 2),
        "4R Why": why,
        "4R Next Trigger": next_trigger,
    }


def _phase4r2_num(value):
    """Convert pandas/numpy values to JSON-safe finite Python floats."""
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def _phase4r2_int(value):
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return None
        return int(value)
    except Exception:
        return None


def _phase4r2_snapshot_payload(result_df):
    """
    Convert one completed scanner result into durable Phase 4R.2 snapshot rows.
    All scored tickers are stored, including BACKGROUND names, because those
    rows are essential for reconstructing how a setup evolved before WATCH.
    """
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        raise RuntimeError("Durable storage is not configured.")

    # One id shared by every ticker in this scanner run.
    scan_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S.%f%z")
    payload = []

    for _, row in result_df.iterrows():
        payload.append(
            {
                "owner_id": cfg["owner_id"],
                "scan_id": scan_id,
                "ticker": str(row.get("Ticker", "")).upper().strip(),
                "stage": str(row.get("4R Stage", "Unavailable")),
                "stage_rank": _phase4r2_int(row.get("4R Stage Rank")) or 0,
                "bullseye_score": _phase4r2_num(row.get("Bullseye 4.0 Score")),
                "gap_to_90": _phase4r2_num(row.get("4R Gap to 90")),
                "readiness": _phase4r2_int(row.get("4R Readiness")),
                "why": str(row.get("4R Why", "")),
                "next_trigger": str(row.get("4R Next Trigger", "")),
                "price": _phase4r2_num(row.get("Price")),
                "accelerator": _phase4r2_num(row.get("4.0 Accelerator")),
                "core_count": _phase4r2_int(row.get("4H Core Count")),
                "momentum_accel": _phase4r2_num(row.get("Momentum Accel")),
                "rel_vol": _phase4r2_num(row.get("Rel Vol")),
                "rs_vs_spy_20d": _phase4r2_num(row.get("RS vs SPY 20D")),
                "rsi": _phase4r2_num(row.get("RSI")),
                "dist_20ma_pct": _phase4r2_num(row.get("Dist 20MA %")),
                "signal_tier": str(row.get("4H Signal Tier", "")),
                "action": str(row.get("4I Action", "")),
                "action_rank": _phase4r2_int(row.get("4I Action Rank")),
                "signal_badges": str(row.get("4H Signal Badges", "")),
                "market_regime": str(row.get("Market Regime", "")),
                "opportunity_state": str(row.get("4R.3 Opportunity State", "NOT READY")),
                "opportunity_icon": str(row.get("4R.3 Opportunity Icon", "⏳")),
                "opportunity_rank": _phase4r2_int(row.get("4R.3 Opportunity Rank")) or 0,
                "opportunity_why": str(row.get("4R.3 Why", "")),
                "opportunity_next_trigger": str(row.get("4R.3 Next Trigger", "")),
            }
        )

    return scan_id, payload


def _phase4r2_save_snapshot(result_df):
    scan_id, payload = _phase4r2_snapshot_payload(result_df)
    if not payload:
        return {"saved": 0, "scan_id": scan_id}

    _phase4q5_request(
        "POST",
        "bullseye_early_warning_snapshots",
        payload=payload,
        prefer="return=minimal",
    )
    return {"saved": len(payload), "scan_id": scan_id}


def _phase4r2_load_history(ticker, limit=100):
    cfg = _phase4q5_storage_config()
    if not cfg["configured"]:
        raise RuntimeError("Durable storage is not configured.")

    ticker = str(ticker).upper().strip()
    if not ticker:
        return []

    params = {
        "select": (
            "scanned_at,scan_id,ticker,stage,stage_rank,bullseye_score,gap_to_90,"
            "readiness,price,accelerator,core_count,momentum_accel,rel_vol,"
            "rs_vs_spy_20d,rsi,dist_20ma_pct,signal_tier,action,market_regime,"
            "why,next_trigger,opportunity_state,opportunity_icon,opportunity_rank,"
            "opportunity_why,opportunity_next_trigger"
        ),
        "owner_id": f"eq.{cfg['owner_id']}",
        "ticker": f"eq.{ticker}",
        "order": "scanned_at.asc",
        "limit": str(int(limit)),
    }
    return _phase4q5_request(
        "GET",
        "bullseye_early_warning_snapshots",
        params=params,
    ) or []


def _phase4r2_history_frame(records):
    if not records:
        return pd.DataFrame()

    hist = pd.DataFrame(records)
    if "scanned_at" in hist.columns:
        ts = pd.to_datetime(hist["scanned_at"], errors="coerce", utc=True)
        hist["Scanned ET"] = ts.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %H:%M:%S")

    rename = {
        "stage": "Stage",
        "bullseye_score": "Score",
        "gap_to_90": "Gap to 90",
        "readiness": "Readiness",
        "price": "Price",
        "accelerator": "Accelerator",
        "core_count": "Core Count",
        "momentum_accel": "Momentum Accel",
        "rel_vol": "Rel Vol",
        "rs_vs_spy_20d": "RS vs SPY 20D",
        "rsi": "RSI",
        "dist_20ma_pct": "Dist 20MA %",
        "signal_tier": "4H Tier",
        "action": "4I Action",
        "market_regime": "Market Regime",
        "why": "Why",
        "next_trigger": "Next Trigger",
        "opportunity_state": "Opportunity State",
        "opportunity_icon": "Opportunity Icon",
        "opportunity_rank": "Opportunity Rank",
        "opportunity_why": "Opportunity Why",
        "opportunity_next_trigger": "Opportunity Next Trigger",
    }
    return hist.rename(columns=rename)


def backtest_symbol(df, spy, ticker, lookback_days=120, step=5):
    """Re-score historical snapshots and measure forward 1–4 week returns."""
    rows = []
    max_forward = 20
    start = max(220, len(df) - lookback_days - max_forward)

    for i in range(start, len(df) - max_forward, step):
        date = df.index[i]
        stock_hist = df.iloc[:i + 1].copy()
        spy_hist = spy.loc[:date].copy()

        if len(stock_hist) < 220 or len(spy_hist) < 220:
            continue

        try:
            scored = score_stock(stock_hist, spy_hist)
            entry = float(df["Close"].iloc[i])
            row = {
                "Date": date,
                "Ticker": ticker,
                "Bullseye Score": scored["Score"],
                "Opportunity Score": scored["Opportunity Score"],
                "Opportunity Rating": scored["Opportunity Rating"],
                "Experimental 3.0 Score": scored["Experimental 3.0 Score"],
                "Experimental 3.0 Rating": scored["Experimental 3.0 Rating"],
                "Bullseye 4.0 Score": scored["Bullseye 4.0 Score"],
                "Bullseye 4.0 Rating": scored["Bullseye 4.0 Rating"],
                "4.0 Accelerator": scored["4.0 Accelerator"],
                "Bullseye 4B1 Score": scored["Bullseye 4B1 Score"],
                "Bullseye 4B2 Score": scored["Bullseye 4B2 Score"],
                "Bullseye 4B3 Score": scored["Bullseye 4B3 Score"],
                "4B1 Accelerator": scored["4B1 Accelerator"],
                "4B2 Accelerator": scored["4B2 Accelerator"],
                "4B3 Accelerator": scored["4B3 Accelerator"],
                "Beta 120D": scored["Beta 120D"],
                "Momentum": scored["Momentum"],
                "Volume": scored["Volume"],
                "Relative Strength": scored["Relative Strength"],
                "Technical": scored["Technical"],
                "Setup Quality": scored["Setup Quality"],
                "Extension Penalty": scored["Extension Penalty"],
                "RSI": scored["RSI"],
                "Dist 20MA %": scored["Dist 20MA %"],
                "Momentum Accel": scored["Momentum Accel"],
                "Market Regime": scored["Market Regime"],
                "Risk/Liquidity": scored["Risk/Liquidity"],
            }
            for days in (5, 10, 15, 20):
                future = float(df["Close"].iloc[i + days])
                row[f"{days}D Forward %"] = round(pct(future, entry), 2)
            rows.append(row)
        except Exception:
            continue

    return rows


def run_period_validation(data, tickers, periods, step):
    """Run the same historical signal study across multiple lookback windows."""
    period_rows = []
    spy = one_symbol(data, "SPY")
    if spy is None:
        return pd.DataFrame()

    signal_names = [
        "Bullseye Score", "Opportunity Score", "Experimental 3.0 Score", "Momentum", "Volume",
        "Relative Strength", "Technical", "Setup Quality",
        "Extension Penalty", "RSI", "Dist 20MA %", "Momentum Accel",
        "Market Regime", "Risk/Liquidity",
    ]

    for label, lookback_days in periods:
        bt_rows = []
        for t in tickers:
            df = one_symbol(data, t)
            if df is None:
                continue
            bt_rows.extend(
                backtest_symbol(
                    df,
                    spy,
                    t,
                    lookback_days=lookback_days,
                    step=step,
                )
            )

        if not bt_rows:
            continue

        bt = pd.DataFrame(bt_rows)

        for signal in signal_names:
            temp = bt[[signal, "20D Forward %"]].dropna().copy()
            if len(temp) < 50 or temp[signal].nunique() < 3:
                continue

            try:
                temp["Q"] = pd.qcut(temp[signal], q=5, duplicates="drop")
            except Exception:
                continue

            grp = temp.groupby("Q", observed=True).agg(
                Samples=(signal, "count"),
                Avg_20D=("20D Forward %", "mean"),
                Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
            ).reset_index(drop=True)

            if len(grp) < 2:
                continue

            low = grp.iloc[0]
            high = grp.iloc[-1]

            period_rows.append({
                "Period": label,
                "Signal": signal,
                "Samples": len(temp),
                "Low 20% Avg 20D": round(low["Avg_20D"], 2),
                "High 20% Avg 20D": round(high["Avg_20D"], 2),
                "High-Low Spread": round(high["Avg_20D"] - low["Avg_20D"], 2),
                "Low 20% Win %": round(low["Win_20D"], 2),
                "High 20% Win %": round(high["Win_20D"], 2),
                "High 20% Hit 5%": round(high["Hit_5pct_20D"], 2),
            })

    return pd.DataFrame(period_rows)


def point_in_time_backtest_symbol(df, spy, ticker, lookback_days=1260, step=20):
    """Historical snapshots with point-in-time stock characteristics."""
    rows = []
    max_forward = 20
    start = max(220, len(df) - lookback_days - max_forward)

    spy_returns_full = spy["Close"].pct_change()

    for i in range(start, len(df) - max_forward, step):
        date = df.index[i]
        stock_hist = df.iloc[:i + 1].copy()
        spy_hist = spy.loc[:date].copy()

        if len(stock_hist) < 220 or len(spy_hist) < 220:
            continue

        try:
            scored = score_stock(stock_hist, spy_hist)

            c = stock_hist["Close"]
            v = stock_hist["Volume"]
            returns = c.pct_change()

            # Point-in-time beta using trailing 120 trading days.
            stock_ret = returns.tail(120).dropna()
            spy_ret = spy_returns_full.loc[:date].tail(120).dropna()
            common = stock_ret.index.intersection(spy_ret.index)

            beta = np.nan
            if len(common) >= 60:
                sr = stock_ret.loc[common]
                pr = spy_ret.loc[common]
                spy_var = float(pr.var())
                if spy_var > 0:
                    beta = float(sr.cov(pr) / spy_var)

            ann_vol = float(returns.tail(60).std() * math.sqrt(252) * 100)
            avg_dollar_vol = float((c * v).tail(60).mean())

            ret_20 = pct(float(c.iloc[-1]), float(c.iloc[-21]))
            ret_60 = pct(float(c.iloc[-1]), float(c.iloc[-61]))
            ret_120 = pct(float(c.iloc[-1]), float(c.iloc[-121]))

            ma20 = float(c.rolling(20).mean().iloc[-1])
            ma50 = float(c.rolling(50).mean().iloc[-1])
            ma200 = float(c.rolling(200).mean().iloc[-1])

            dist20 = pct(float(c.iloc[-1]), ma20)
            dist50 = pct(float(c.iloc[-1]), ma50)

            entry = float(df["Close"].iloc[i])

            row = {
                "Date": date,
                "Ticker": ticker,
                "Experimental 3.0 Score": scored["Experimental 3.0 Score"],
                "Experimental 3.0 Rating": scored["Experimental 3.0 Rating"],
                "Bullseye 4.0 Score": scored["Bullseye 4.0 Score"],
                "Bullseye 4.0 Rating": scored["Bullseye 4.0 Rating"],
                "4.0 Accelerator": scored["4.0 Accelerator"],
                "Bullseye 4B1 Score": scored["Bullseye 4B1 Score"],
                "Bullseye 4B2 Score": scored["Bullseye 4B2 Score"],
                "Bullseye 4B3 Score": scored["Bullseye 4B3 Score"],
                "4B1 Accelerator": scored["4B1 Accelerator"],
                "4B2 Accelerator": scored["4B2 Accelerator"],
                "4B3 Accelerator": scored["4B3 Accelerator"],
                "Bullseye Score": scored["Score"],
                "Opportunity Score": scored["Opportunity Score"],
                "Relative Strength": scored["Relative Strength"],
                "RSI": scored["RSI"],
                "Momentum": scored["Momentum"],
                "Volume": scored["Volume"],
                "Technical": scored["Technical"],
                "Extension Penalty": scored["Extension Penalty"],
                "Market Regime": scored["Market Regime"],
                "Beta vs SPY": beta,
                "Ann Vol %": ann_vol,
                "Avg $ Volume 60D ($M)": avg_dollar_vol / 1_000_000,
                "20D Return %": ret_20,
                "60D Return %": ret_60,
                "120D Return %": ret_120,
                "Dist 20MA %": dist20,
                "Dist 50MA %": dist50,
                "Above 20/50/200": bool(c.iloc[-1] > ma20 > ma50 > ma200),
            }

            for days in (5, 10, 15, 20):
                future = float(df["Close"].iloc[i + days])
                row[f"{days}D Forward %"] = round(pct(future, entry), 2)

            rows.append(row)
        except Exception:
            continue

    return rows


def build_trade_plan(df, scored_row):
    """Create swing-trade reference entry, invalidation, and targets from current structure."""
    c = df["Close"]
    h = df["High"]
    l = df["Low"]

    if len(df) < 60:
        return None

    last = float(c.iloc[-1])
    prev_close = c.shift(1)
    tr = pd.concat(
        [
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1])

    if not np.isfinite(atr14) or atr14 <= 0:
        return None

    ma20 = float(c.rolling(20).mean().iloc[-1])
    ma50 = float(c.rolling(50).mean().iloc[-1])
    high20 = float(h.tail(20).max())
    low5 = float(l.tail(5).min())
    low10 = float(l.tail(10).min())

    rsi = float(scored_row.get("RSI", np.nan))
    dist20 = float(scored_row.get("Dist 20MA %", np.nan))
    rel_vol = float(scored_row.get("Rel Vol", np.nan))

    near_breakout = last >= high20 * 0.985
    extended = (pd.notna(dist20) and dist20 >= 15) or (pd.notna(rsi) and rsi >= 80)

    if extended:
        entry_mode = "Wait for pullback"
    elif near_breakout and (pd.isna(rel_vol) or rel_vol >= 0.9):
        entry_mode = "Breakout / continuation watch"
    else:
        entry_mode = "Pullback / continuation watch"

    # Preserve the original pullback-entry behavior.
    entry_high = last - 0.25 * atr14
    entry_low = last - 0.75 * atr14
    entry_low = max(entry_low, ma20 * 0.995)
    entry_high = max(entry_high, entry_low)

    if last <= ma20 * 1.03:
        entry_low = max(last - 0.50 * atr14, ma20 * 0.995)
        entry_high = last - 0.10 * atr14

    breakout_entry = high20 + 0.10 * atr14
    entry_mid = (entry_low + entry_high) / 2

    # Phase 4N.1 invalidation hierarchy:
    # 1) nearby swing structure,
    # 2) 20MA / ATR structure,
    # 3) ATR-aware max-risk guardrail for a 1–4 week swing.
    swing_ref = low5 - 0.20 * atr14
    ma_ref = ma20 - 0.60 * atr14
    ten_day_ref = low10 - 0.10 * atr14

    technical_candidates = [
        x for x in [swing_ref, ma_ref, ten_day_ref]
        if np.isfinite(x) and x < entry_low
    ]

    # Prefer the nearest valid technical level below entry, not the deepest one.
    technical_stop = max(technical_candidates) if technical_candidates else entry_low - 1.25 * atr14

    # Guardrail scales with ATR but caps routine swing risk.
    atr_risk_pct = (1.35 * atr14 / entry_mid) * 100 if entry_mid else np.nan
    max_risk_pct = float(np.clip(atr_risk_pct, 4.0, 8.0))
    guardrail_stop = entry_mid * (1 - max_risk_pct / 100)

    # Stop must remain below entry, while avoiding unnecessarily deep invalidation.
    invalidation = max(technical_stop, guardrail_stop)
    invalidation = min(invalidation, entry_low - 0.25 * atr14)

    risk_per_share = max(entry_mid - invalidation, 0.01)
    risk_pct = (risk_per_share / entry_mid) * 100 if entry_mid else np.nan

    target_1r = entry_mid + risk_per_share
    target_2r = entry_mid + 2 * risk_per_share
    target_3r = entry_mid + 3 * risk_per_share

    breakout_chase_pct = ((breakout_entry / last) - 1) * 100 if last else np.nan

    if risk_pct <= 4.5:
        risk_label = "Tight"
    elif risk_pct <= 6.5:
        risk_label = "Normal"
    elif risk_pct <= 8:
        risk_label = "Wide"
    else:
        risk_label = "Too wide"

    return {
        "Entry Mode": entry_mode,
        "Current Price": round(last, 2),
        "ATR14": round(atr14, 2),
        "20MA": round(ma20, 2),
        "50MA": round(ma50, 2),
        "Pullback Entry Low": round(entry_low, 2),
        "Pullback Entry High": round(entry_high, 2),
        "Breakout Reference": round(breakout_entry, 2),
        "Invalidation Reference": round(invalidation, 2),
        "Risk / Share": round(risk_per_share, 2),
        "Risk %": round(risk_pct, 2),
        "Risk Label": risk_label,
        "Target 1R": round(target_1r, 2),
        "Target 2R": round(target_2r, 2),
        "Target 3R": round(target_3r, 2),
        "Breakout Distance %": round(breakout_chase_pct, 2),
    }




def build_candidate_investigation(scored, trade_plan, current_mark):
    """
    Phase 4Q.7 pre-entry interpretation layer.

    Uses the already-validated Bullseye score and Phase 4N trade-plan references.
    It does not alter scoring, targets, invalidation, or live-position management.
    """
    if not scored or not trade_plan or not np.isfinite(current_mark) or current_mark <= 0:
        return {
            "Candidate Action": "Unavailable",
            "Action Reason": "Insufficient current market or trade-plan data.",
            "Entry Distance %": np.nan,
            "Entry Zone Position": "Unavailable",
        }

    score = float(scored.get("Bullseye 4.0 Score", np.nan))
    action_rank = int(scored.get("4I Action Rank", 0) or 0)
    rsi = float(scored.get("RSI", np.nan))
    dist20 = float(scored.get("Dist 20MA %", np.nan))

    entry_low = float(trade_plan.get("Pullback Entry Low", np.nan))
    entry_high = float(trade_plan.get("Pullback Entry High", np.nan))
    breakout = float(trade_plan.get("Breakout Reference", np.nan))
    invalidation = float(trade_plan.get("Invalidation Reference", np.nan))
    atr = float(trade_plan.get("ATR14", np.nan))
    entry_mode = str(trade_plan.get("Entry Mode", ""))

    entry_mid = (
        (entry_low + entry_high) / 2
        if np.isfinite(entry_low) and np.isfinite(entry_high)
        else np.nan
    )
    entry_distance_pct = (
        ((current_mark / entry_mid) - 1) * 100
        if np.isfinite(entry_mid) and entry_mid > 0
        else np.nan
    )

    if np.isfinite(invalidation) and current_mark <= invalidation:
        action = "Setup Invalidated"
        reason = (
            f"Current mark ${current_mark:,.2f} is at or below Bullseye's "
            f"${invalidation:,.2f} technical invalidation reference."
        )
        zone_position = "Below invalidation"

    elif pd.notna(score) and score < 60:
        action = "Low Priority / Avoid"
        reason = f"Bullseye 4.0 score is {score:.1f}, below the 60-point Watch threshold."
        zone_position = "Low-priority setup"

    elif np.isfinite(entry_low) and np.isfinite(entry_high) and entry_low <= current_mark <= entry_high:
        action = "Entry Zone"
        reason = (
            f"Current mark ${current_mark:,.2f} is inside the Bullseye pullback entry zone "
            f"${entry_low:,.2f}–${entry_high:,.2f}."
        )
        zone_position = "Inside pullback entry zone"

    elif np.isfinite(entry_low) and current_mark < entry_low:
        distance = entry_low - current_mark
        if np.isfinite(atr) and atr > 0 and distance <= 0.50 * atr:
            action = "Approaching Entry"
            reason = (
                f"Current mark is below the planned entry zone but within 0.5 ATR of "
                f"the ${entry_low:,.2f} lower entry boundary."
            )
            zone_position = "Just below entry zone"
        else:
            action = "Watch / Below Entry Zone"
            reason = (
                f"Current mark ${current_mark:,.2f} remains below the planned "
                f"${entry_low:,.2f}–${entry_high:,.2f} entry zone."
            )
            zone_position = "Below entry zone"

    else:
        extended = (
            "Wait for pullback" in entry_mode
            or (pd.notna(dist20) and dist20 >= 12)
            or (pd.notna(rsi) and rsi >= 78)
        )

        if extended:
            action = "Extended — Wait"
            reason = (
                "Price is above the pullback entry area and the existing Bullseye "
                "extension/timing signals favor waiting rather than chasing."
            )
            zone_position = "Above entry zone / extended"

        elif np.isfinite(breakout) and current_mark >= breakout:
            breakout_distance = current_mark - breakout
            if np.isfinite(atr) and atr > 0 and breakout_distance <= 0.50 * atr:
                action = "Breakout Area"
                reason = (
                    f"Current mark is near/above the ${breakout:,.2f} breakout reference "
                    "without yet meeting the conservative extension filter."
                )
                zone_position = "Near breakout reference"
            else:
                action = "Extended — Wait"
                reason = (
                    f"Current mark is materially above the ${breakout:,.2f} breakout reference; "
                    "avoid chasing and wait for a better risk/reward location."
                )
                zone_position = "Above breakout reference"

        else:
            action = "Watch / Above Pullback Zone"
            reason = (
                "Price is above the preferred pullback entry zone but has not triggered "
                "the conservative extension classification."
            )
            zone_position = "Above pullback entry zone"

    # High-conviction score does not override entry discipline; it is shown separately.
    if action_rank >= 1 and action in ("Watch / Below Entry Zone", "Watch / Above Pullback Zone"):
        reason += f" Signal tier remains {scored.get('4H Signal Tier', 'active')}."

    return {
        "Candidate Action": action,
        "Action Reason": reason,
        "Entry Distance %": round(float(entry_distance_pct), 2) if pd.notna(entry_distance_pct) else np.nan,
        "Entry Zone Position": zone_position,
    }


def build_phase4r3_opportunity_state(scored, trade_plan, current_mark):
    """
    Phase 4R.3 — Entry Timing / Opportunity State.

    Read-only timing layer. It does NOT alter Bullseye 4.0 scoring, 4R.1 stage,
    trade-plan levels, or live-position management math. It answers a separate
    question: "Bullseye likes this setup — what should I do with it right now?"

    States are intentionally conservative:
      👀 APPROACHING — get ready; location/setup is moving toward actionability
      🎯 ACTIONABLE  — qualified setup currently inside preferred entry zone
      🚀 BREAKOUT    — qualified setup challenging/clearing breakout with support
      ⚠️ EXTENDED    — setup may remain strong, but current location favors waiting
      ⏳ NOT READY   — setup/location is not actionable yet
    """
    icons = {
        "APPROACHING": "👀",
        "ACTIONABLE": "🎯",
        "BREAKOUT": "🚀",
        "EXTENDED": "⚠️",
        "NOT READY": "⏳",
    }
    ranks = {
        "NOT READY": 0,
        "APPROACHING": 1,
        "EXTENDED": 2,
        "BREAKOUT": 3,
        "ACTIONABLE": 4,
    }

    def pack(state, why, next_trigger):
        icon = icons[state]
        return {
            "4R.3 Opportunity State": state,
            "4R.3 Opportunity Icon": icon,
            "4R.3 Opportunity Signal": f"{icon} {state}",
            "4R.3 Opportunity Rank": ranks[state],
            "4R.3 Why": why,
            "4R.3 Next Trigger": next_trigger,
        }

    if not scored or not trade_plan:
        return pack(
            "NOT READY",
            "Bullseye does not yet have enough valid setup/trade-plan data to classify entry timing.",
            "Wait for a valid Bullseye setup and technical trade plan.",
        )

    try:
        current_mark = float(current_mark)
    except Exception:
        current_mark = np.nan
    if not np.isfinite(current_mark) or current_mark <= 0:
        return pack(
            "NOT READY",
            "A valid current price is unavailable, so entry timing cannot be classified safely.",
            "Refresh market data and re-evaluate the candidate.",
        )

    stage = str(scored.get("4R Stage", "BACKGROUND")).upper().strip()
    score = float(scored.get("Bullseye 4.0 Score", np.nan))
    rel_vol = float(scored.get("Rel Vol", np.nan))
    accel = float(scored.get("4.0 Accelerator", np.nan))
    mom_accel = float(scored.get("Momentum Accel", np.nan))
    core = int(scored.get("4H Core Count", 0) or 0)
    rsi = float(scored.get("RSI", np.nan))
    dist20 = float(scored.get("Dist 20MA %", np.nan))

    entry_low = float(trade_plan.get("Pullback Entry Low", np.nan))
    entry_high = float(trade_plan.get("Pullback Entry High", np.nan))
    breakout = float(trade_plan.get("Breakout Reference", np.nan))
    invalidation = float(trade_plan.get("Invalidation Reference", np.nan))
    atr = float(trade_plan.get("ATR14", np.nan))
    entry_mode = str(trade_plan.get("Entry Mode", ""))

    stage_ready = stage in {"WATCH", "DEVELOPING", "QUALIFIED"}
    qualified = stage == "QUALIFIED" or (np.isfinite(score) and score >= 90)

    if np.isfinite(invalidation) and current_mark <= invalidation:
        return pack(
            "NOT READY",
            f"Current price ${current_mark:,.2f} is at/below the ${invalidation:,.2f} invalidation reference.",
            "Require a fresh Bullseye setup before considering a new entry.",
        )

    if not stage_ready:
        return pack(
            "NOT READY",
            f"4R stage is {stage or 'BACKGROUND'}; the setup has not reached Bullseye's near-qualification timing pool.",
            str(scored.get("4R Next Trigger", "Wait for stronger setup confirmation.")),
        )

    # A qualified setup inside the preferred pullback entry zone is the cleanest
    # actionable condition. WATCH/DEVELOPING names in the same location remain
    # APPROACHING until the setup itself qualifies.
    if np.isfinite(entry_low) and np.isfinite(entry_high) and entry_low <= current_mark <= entry_high:
        if qualified:
            return pack(
                "ACTIONABLE",
                f"Qualified setup with price ${current_mark:,.2f} inside the preferred Bullseye entry zone ${entry_low:,.2f}–${entry_high:,.2f}.",
                "Maintain qualification and defined risk; use the planned invalidation reference for risk control.",
            )
        return pack(
            "APPROACHING",
            f"Price is already inside the preferred entry zone, but the setup is still {stage} rather than QUALIFIED.",
            f"Need Bullseye qualification at 90+ while price remains near ${entry_low:,.2f}–${entry_high:,.2f}.",
        )

    # Below the entry zone: surface it only when close enough to matter.
    if np.isfinite(entry_low) and current_mark < entry_low:
        distance = entry_low - current_mark
        if np.isfinite(atr) and atr > 0 and distance <= 0.75 * atr:
            return pack(
                "APPROACHING",
                f"Price is just below the entry zone and within 0.75 ATR of the ${entry_low:,.2f} lower boundary.",
                f"Watch for price to reclaim/enter ${entry_low:,.2f}–${entry_high:,.2f} while setup quality holds.",
            )
        return pack(
            "NOT READY",
            f"Price ${current_mark:,.2f} remains below the planned entry zone and is not yet close enough for an entry-timing alert.",
            f"Watch for an approach toward the ${entry_low:,.2f} lower entry boundary.",
        )

    breakout_support = (
        (np.isfinite(rel_vol) and rel_vol >= 0.90)
        or (np.isfinite(accel) and accel >= 8)
        or (np.isfinite(mom_accel) and mom_accel > 0)
        or core >= 2
    )

    materially_above_breakout = (
        np.isfinite(breakout)
        and np.isfinite(atr)
        and atr > 0
        and current_mark > breakout + 0.50 * atr
    )
    extension_caution = (
        "Wait for pullback" in entry_mode
        or (np.isfinite(dist20) and dist20 >= 12)
        or (np.isfinite(rsi) and rsi >= 78)
        or materially_above_breakout
    )

    # Preserve entry discipline: an extended setup does not become a breakout
    # chase simply because price crossed the breakout reference.
    if extension_caution:
        details = []
        if "Wait for pullback" in entry_mode:
            details.append("trade plan favors a pullback")
        if np.isfinite(dist20) and dist20 >= 12:
            details.append(f"price is {dist20:.1f}% above the 20MA")
        if np.isfinite(rsi) and rsi >= 78:
            details.append(f"RSI is {rsi:.1f}")
        if materially_above_breakout:
            details.append("price is >0.5 ATR above breakout")
        detail_text = "; ".join(details) if details else "extension filters are active"
        return pack(
            "EXTENDED",
            f"Setup may remain strong, but current location favors waiting rather than chasing: {detail_text}.",
            f"Prefer a pullback toward ${entry_high:,.2f} or a new consolidation that restores favorable risk/reward."
            if np.isfinite(entry_high) else "Wait for a new consolidation/pullback before considering entry.",
        )

    # A clean breakout needs qualification plus at least one independent support
    # signal. This prevents a high score alone from turning every cross into 🚀.
    if np.isfinite(breakout) and current_mark >= breakout:
        near_breakout = (
            not np.isfinite(atr)
            or atr <= 0
            or current_mark <= breakout + 0.50 * atr
        )
        if qualified and breakout_support and near_breakout:
            return pack(
                "BREAKOUT",
                f"Qualified setup is challenging/clearing the ${breakout:,.2f} breakout reference with supporting momentum/volume/core evidence.",
                "Look for the breakout to hold without becoming extended; avoid chasing if price stretches >0.5 ATR above the reference.",
            )
        return pack(
            "NOT READY",
            f"Price is at/above the ${breakout:,.2f} breakout reference, but Bullseye does not yet have the full qualification/support combination for a breakout entry.",
            "Require qualification plus supporting volume/momentum/core confirmation, or wait for a pullback into the preferred entry area.",
        )

    # Above pullback zone but below breakout: useful alert territory, provided the
    # setup is not already extended.
    if np.isfinite(entry_high) and current_mark > entry_high:
        trigger = (
            f"Watch ${breakout:,.2f} for a supported breakout, or a pullback into ${entry_low:,.2f}–${entry_high:,.2f}."
            if np.isfinite(breakout) and np.isfinite(entry_low)
            else "Watch for either a supported breakout or a pullback into the preferred entry zone."
        )
        return pack(
            "APPROACHING",
            "Price is above the preferred pullback zone but has not become extended or completed a supported breakout.",
            trigger,
        )

    return pack(
        "NOT READY",
        "Current setup/location does not match an actionable Bullseye 4R.3 entry condition.",
        "Continue watching for an entry-zone approach or supported breakout.",
    )


def build_corr_clusters(price_data, tickers, corr_threshold=0.70, lookback=60):
    """Build simple connected correlation clusters from trailing daily returns."""
    series = {}
    for t in tickers:
        df = one_symbol(price_data, t)
        if df is None or len(df) < max(30, lookback // 2):
            continue
        s = df["Close"].pct_change().dropna().tail(lookback)
        if len(s) >= 20:
            series[t] = s

    if not series:
        return {}, pd.DataFrame()

    returns = pd.concat(series, axis=1).dropna(how="all")
    corr = returns.corr(min_periods=15)

    parent = {t: t for t in corr.columns}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            val = corr.loc[a, b]
            if pd.notna(val) and val >= corr_threshold:
                union(a, b)

    roots = {}
    for t in cols:
        r = find(t)
        roots.setdefault(r, []).append(t)

    cluster_map = {}
    for idx, members in enumerate(sorted(roots.values(), key=lambda x: sorted(x)[0]), start=1):
        label = f"C{idx}"
        for t in members:
            cluster_map[t] = label

    return cluster_map, corr


def build_trade_management_plan(df, trade_plan, trim_pct=50, trail_start_r=1.5, trail_atr=1.0):
    """Create a reference post-entry management plan from the frozen Phase 4N.1 trade plan."""
    if trade_plan is None or len(df) < 60:
        return None

    entry = (float(trade_plan["Pullback Entry Low"]) + float(trade_plan["Pullback Entry High"])) / 2
    stop = float(trade_plan["Invalidation Reference"])
    atr = float(trade_plan["ATR14"])
    current = float(trade_plan["Current Price"])

    risk = max(entry - stop, 0.01)
    current_r = (current - entry) / risk
    t1 = float(trade_plan["Target 1R"])
    t2 = float(trade_plan["Target 2R"])
    t3 = float(trade_plan["Target 3R"])

    breakeven_trigger = entry + risk
    trail_trigger = entry + float(trail_start_r) * risk
    trailing_stop = max(stop, current - float(trail_atr) * atr)

    if current <= stop:
        action = "Exit"
        reason = "Price is at or below the invalidation reference."
    elif current >= t2:
        action = "Trim / Trail"
        reason = "Trade has reached 2R or better; protect gains and trail the remainder."
    elif current >= t1:
        action = "Trim"
        reason = f"Target 1 reached; consider taking {int(trim_pct)}% partial profit."
    elif current >= trail_trigger:
        action = "Hold / Trail"
        reason = f"Profit exceeds {trail_start_r:.1f}R; trail using roughly {trail_atr:.2f} ATR."
    elif current >= breakeven_trigger:
        action = "Hold / Protect"
        reason = "Trade is at least 1R in profit; consider raising risk toward breakeven."
    else:
        action = "Hold"
        reason = "Trade remains between entry and the first profit-protection threshold."

    return {
        "Entry Reference": round(entry, 2),
        "Initial Stop": round(stop, 2),
        "Risk / Share": round(risk, 2),
        "Current Price": round(current, 2),
        "Current R": round(current_r, 2),
        "Target 1": round(t1, 2),
        "Target 2": round(t2, 2),
        "Target 3": round(t3, 2),
        "R:R to T1": round((t1 - entry) / risk, 2),
        "R:R to T2": round((t2 - entry) / risk, 2),
        "R:R to T3": round((t3 - entry) / risk, 2),
        "Partial Profit %": int(trim_pct),
        "Breakeven Trigger": round(breakeven_trigger, 2),
        "Trail Trigger": round(trail_trigger, 2),
        "ATR Trail Ref": round(trailing_stop, 2),
        "Management Action": action,
        "Management Reason": reason,
    }



# Phase 4Q.1 persistent actual-position inputs.
_phase4q1_defaults = {
    "phase4q1_state_key": "Candidate / Watching",
    "phase4q1_ticker_key": "",
    "phase4q1_entry_key": 0.0,
    "phase4q1_initial_shares_key": 0.0,
    "phase4q1_remaining_shares_key": 0.0,
    "phase4q1_realized_pl_key": 0.0,
    "phase4q1_initial_stop_key": 0.0,
    "phase4q1_actual_stop_key": 0.0,
    "phase4q3_test_mode_key": False,
    "phase4q3_test_mark_key": 0.0,
    "phase4q4_live_state": {},
    "phase4q4_test_state": None,
    "phase4q1_view_active": False,
    "phase4q5_last_loaded": None,
    "phase4q5_last_saved": None,
    "phase4q5_last_message": "",
    "phase4q8_selected_candidate": "",
    "phase4q8_message": "",
    "phase4q8_promotion_active": False,
    "phase4q8_promotion_ticker": "",
    "phase4q8_promotion_snapshot": {},
    "phase4q5_delete_confirm_key": "",
    "phase4q8_clear_selected_candidate_on_next_run": False,
    "phase4q9_close_confirm_key": "",
    "phase4q9_message": "",
    "phase4q9_clear_position_on_next_run": False,
    "phase4r2a_investigate_ticker": "",
    "phase4r2f_account_size_loaded": False,
    "phase4r2f_account_size_message": "",
}
for _k, _v in _phase4q1_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Phase 4R.2A: apply scanner → Candidate Investigation selection before
# Phase 4Q.1 widgets are instantiated. This preserves Streamlit widget-state safety.
_phase4r2a_requested = str(st.session_state.pop("phase4r2a_investigate_ticker", "")).upper().strip()
if _phase4r2a_requested:
    st.session_state["phase4q1_state_key"] = "Candidate / Watching"
    st.session_state["phase4q1_ticker_key"] = _phase4r2a_requested
    st.session_state["phase4q1_view_active"] = True

# Apply deferred cleanup before the Saved Candidates selectbox is instantiated.
if st.session_state.get("phase4q8_clear_selected_candidate_on_next_run", False):
    st.session_state["phase4q8_selected_candidate"] = ""
    st.session_state["phase4q8_clear_selected_candidate_on_next_run"] = False

# Apply successful closeout cleanup before Phase 4Q.1 widgets are instantiated.
# This avoids StreamlitAPIException from mutating widget-bound keys late in a run.
if st.session_state.get("phase4q9_clear_position_on_next_run", False):
    st.session_state["phase4q1_state_key"] = "Candidate / Watching"
    st.session_state["phase4q1_ticker_key"] = ""
    st.session_state["phase4q1_entry_key"] = 0.0
    st.session_state["phase4q1_initial_shares_key"] = 0.0
    st.session_state["phase4q1_remaining_shares_key"] = 0.0
    st.session_state["phase4q1_realized_pl_key"] = 0.0
    st.session_state["phase4q1_initial_stop_key"] = 0.0
    st.session_state["phase4q1_actual_stop_key"] = 0.0
    st.session_state["phase4q3_test_mode_key"] = False
    st.session_state["phase4q3_test_mark_key"] = 0.0
    st.session_state["phase4q1_view_active"] = False
    st.session_state["phase4q5_last_loaded"] = None
    st.session_state["phase4q9_close_confirm_key"] = ""
    st.session_state["phase4q9_clear_position_on_next_run"] = False


def _clear_phase4q1_inputs():
    """Reset Phase 4Q.1 widget state safely via a Streamlit callback."""
    for _k, _v in _phase4q1_defaults.items():
        st.session_state[_k] = _v


with st.sidebar:
    st.header("Scanner settings")
    scan_universe = st.radio(
        "Scan Universe",
        ["Broad", "Focused", "Custom"],
        index=0,
        horizontal=True,
        help=(
            "Broad scans Bullseye's larger discovery universe. "
            "Focused scans the original development/watch universe. "
            "Custom scans only symbols you enter."
        ),
    )

    if scan_universe == "Broad":
        tickers = sorted(set(x.upper().strip() for x in BROAD_TICKERS if x.strip()))
        st.caption(f"Broad discovery universe • {len(tickers)} tickers")
    elif scan_universe == "Focused":
        tickers = sorted(set(x.upper().strip() for x in DEFAULT_TICKERS if x.strip()))
        st.caption(f"Focused universe • {len(tickers)} tickers")
    else:
        universe_text = st.text_area(
            "Custom tickers (space, comma, or newline separated)",
            " ".join(DEFAULT_TICKERS),
            height=180,
        )
        tickers = sorted(
            set(x.upper().strip() for x in universe_text.replace(",", " ").split() if x.strip())
        )
        st.caption(f"Custom universe • {len(tickers)} tickers")

    run = st.button("🔎 Run scanner", type="primary")

    st.divider()
    with st.expander("📚 Historical Validation", expanded=False):
        backtest_lookback = st.selectbox(
            "Backtest history",
            [252, 504, 756, 1260],
            index=1,
            format_func=lambda x: {
                252: "About 1 year",
                504: "About 2 years",
                756: "About 3 years",
                1260: "About 5 years",
            }[x],
        )
        backtest_step = st.selectbox(
            "Snapshot frequency",
            [5, 10, 20],
            index=0,
            format_func=lambda x: f"Every {x} trading days",
        )
        run_backtest = st.button("🧪 Run backtest")
        run_multi_period = st.button("📚 Run 1Y/2Y/3Y/5Y validation")
        run_walk_forward = st.button("🚶 Run 3D walk-forward test")
        run_stress_test = st.button("🧱 Run 3E stress test")
        run_diagnostics = st.button("🧬 Run 3F breadth diagnostics")
        run_point_in_time = st.button("🕰️ Run 3G point-in-time test")
        run_interactions = st.button("🧩 Run 3H interaction test")
        run_robustness = st.button("🛡️ Run 3I robustness test")
        run_phase4a = st.button("🚀 Run 4A prototype test")
        run_phase4b = st.button("🧪 Run 4B accelerator tuning")
        run_phase4c = st.button("🌐 Run 4C broad-universe test")
        run_phase4d = st.button("🎯 Run 4D threshold test")
        run_phase4e = st.button("🔎 Run 4E high-score diagnostics")
        run_phase4f = st.button("🧭 Run 4F confirmation-layer test")
        run_phase4g = st.button("🧱 Run 4G confirmation robustness test")
        run_phase4h = st.button("🏗️ Run 4H signal-architecture test")
        run_phase4i = st.button("🖥️ Run 4I live decision-screen test")
        run_phase4j = st.button("📝 Run 4J forward signal journal")

    with st.expander("📈 Forward Validation", expanded=False):
        journal_upload = st.file_uploader("Upload a saved Phase 4J journal CSV", type=["csv"])
        run_phase4k = st.button("📈 Run 4K journal review")
        run_phase4l = st.button("📊 Run 4L forward performance dashboard")
        run_phase4m = st.button("🎛️ Run 4M live command center")
        st.divider()
        st.caption("Phase 4R.2 Early-Warning Snapshot History")
        phase4r2_history_ticker = st.text_input(
            "Snapshot history ticker",
            value="COIN",
            key="phase4r2_history_ticker",
        ).upper().strip()
        run_phase4r2_history = st.button("🕒 Load 4R.2 snapshot history")
        st.divider()
        st.caption("Phase 4R.5 Candidate Outcome Journal")
        phase4r5_outcome_ticker = st.text_input("Outcome journal ticker (optional)", value="", key="phase4r5_outcome_ticker").upper().strip()
        run_phase4r5_outcomes = st.button("🎯 Load 4R.5 candidate outcomes")

    with st.expander("📍 Position Management", expanded=False):
        run_phase4n = st.button("🧭 Run 4N entry/exit planner")
        run_phase4o = st.button("🧮 Run 4O position-sizing planner")
        with st.expander("⚙️ Advanced 4O/4P Controls", expanded=False):
            st.caption(
                "Account size is durable and reloads from Supabase. "
                "The remaining 4O/4P tuning controls stay hidden during normal operation."
            )

            if not st.session_state.get("phase4r2f_account_size_loaded", False):
                try:
                    st.session_state["phase4r2f_account_size_key"] = _phase4r2f_load_account_size(25000.0)
                except Exception:
                    st.session_state["phase4r2f_account_size_key"] = 25000.0
                st.session_state["phase4r2f_account_size_loaded"] = True

            phase4o_account_size = st.number_input(
                "Account size ($)",
                min_value=0.0,
                step=100.0,
                format="%.2f",
                key="phase4r2f_account_size_key",
                help="Saved durably in Supabase so it survives Streamlit restarts and redeploys.",
            )
            if st.button("💾 Save account size", key="phase4r2f_save_account_size"):
                try:
                    saved_size = _phase4r2f_save_account_size(phase4o_account_size)
                    st.session_state["phase4r2f_account_size_message"] = (
                        f"Account size saved: ${saved_size:,.2f}"
                    )
                    st.success(st.session_state["phase4r2f_account_size_message"])
                except Exception as exc:
                    st.error(f"Account size was not saved: {exc}")

            if st.session_state.get("phase4r2f_account_size_message"):
                st.caption(st.session_state["phase4r2f_account_size_message"])

            st.divider()
            phase4op_manual_tuning = st.checkbox(
                "Enable manual 4O/4P tuning",
                value=False,
                key="phase4op_manual_tuning_key",
            )

            if phase4op_manual_tuning:
                st.caption("Phase 4O sizing inputs")
                phase4o_risk_pct = st.number_input(
                    "Max account risk per trade (%)",
                    min_value=0.25, max_value=2.00, value=0.75,
                    step=0.25, format="%.2f"
                )
                phase4o_max_position_pct = st.number_input(
                    "Max position size (% of account)",
                    min_value=10.0, max_value=100.0, value=25.0,
                    step=5.0, format="%.1f"
                )

                st.caption("Phase 4P portfolio controls")
                phase4p_max_total_risk_pct = st.number_input(
                    "Max combined open risk (%)",
                    min_value=1.0, max_value=6.0, value=3.0,
                    step=0.5, format="%.2f"
                )
                phase4p_corr_threshold = st.number_input(
                    "Correlation alert threshold",
                    min_value=0.50, max_value=0.90, value=0.70,
                    step=0.05, format="%.2f"
                )
                phase4p_max_cluster_risk_pct = st.number_input(
                    "Max risk per correlation cluster (%)",
                    min_value=0.75, max_value=3.0, value=1.5,
                    step=0.25, format="%.2f"
                )

                st.caption("Phase 4Q trade-management preferences")
                phase4q_trim_pct = st.number_input(
                    "Partial profit at Target 1 (%)",
                    min_value=25.0, max_value=75.0, value=50.0,
                    step=1.0, format="%.0f"
                )
                phase4q_trail_start_r = st.number_input(
                    "Start trailing after profit reaches (R)",
                    min_value=1.0, max_value=2.5, value=1.5,
                    step=0.25, format="%.2f"
                )
                phase4q_trail_atr = st.number_input(
                    "Trailing stop distance (ATR)",
                    min_value=0.75, max_value=2.0, value=1.0,
                    step=0.25, format="%.2f"
                )
            else:
                phase4o_risk_pct = 0.75
                phase4o_max_position_pct = 25.0
                phase4p_max_total_risk_pct = 3.0
                phase4p_corr_threshold = 0.70
                phase4p_max_cluster_risk_pct = 1.5
                phase4q_trim_pct = 50.0
                phase4q_trail_start_r = 1.5
                phase4q_trail_atr = 1.0
                st.caption("Manual tuning OFF — validated/default 4O/4P/4Q values are active.")

        run_phase4p = st.button("🧩 Run 4P portfolio-risk planner")
        run_phase4q = st.button("🧠 Run 4Q trade-management planner")
        run_phase4q1 = st.button("📍 Run 4Q.1 position-state manager")

        st.caption("Phase 4Q.1 actual-position inputs")
        phase4q1_state = st.selectbox(
            "Position state",
            ["Candidate / Watching", "Entered / Live Position", "Closed Trade"],
            key="phase4q1_state_key",
        )
        phase4q1_ticker = st.text_input(
            "Position ticker",
            placeholder="e.g. LLY",
            key="phase4q1_ticker_key",
        ).upper().strip()

        st.divider()
        if phase4q1_state == "Entered / Live Position":
            st.markdown("### 📌 Live Position Entry / Management")
            st.caption("Enter the actual trade details you own. These fields are separate from Closed Trades History.")
            phase4q1_entry = st.number_input(
                "Actual average entry price per share ($)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="phase4q1_entry_key",
            )
            phase4q1_initial_shares = st.number_input(
                "Initial shares",
                min_value=0.0,
                step=0.00001,
                format="%.5f",
                key="phase4q1_initial_shares_key",
            )
            phase4q1_remaining_shares = st.number_input(
                "Shares currently remaining",
                min_value=0.0,
                step=0.00001,
                format="%.5f",
                key="phase4q1_remaining_shares_key",
            )
            phase4q1_realized_pl = st.number_input(
                "Realized P/L so far ($)",
                step=10.0,
                format="%.2f",
                key="phase4q1_realized_pl_key",
            )
            phase4q1_initial_stop = st.number_input(
                "Original stop when trade was opened ($, 0 = unknown)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="phase4q1_initial_stop_key",
            )
            phase4q1_actual_stop = st.number_input(
                "Current stop for remaining shares ($, 0 = use Bullseye)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="phase4q1_actual_stop_key",
            )

            st.caption("Phase 4Q.3 management-state transition test")
            phase4q3_test_mode = st.checkbox(
                "Enable simulated Position Mark",
                key="phase4q3_test_mode_key",
                help="Testing only. Overrides the live Position Mark for 4Q.1/4Q.2 calculations; never changes Bullseye scoring or market data.",
            )
            phase4q3_test_mark = st.number_input(
                "Simulated Position Mark ($)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="phase4q3_test_mark_key",
                disabled=not phase4q3_test_mode,
            )

            st.button(
                "Clear Phase 4Q.1 position inputs",
                on_click=_clear_phase4q1_inputs,
            )

        elif phase4q1_state == "Candidate / Watching":
            st.markdown("### 👀 Candidate / Watching")
            st.caption("Pre-entry investigation only. Live-position ownership fields remain hidden.")
            phase4q1_entry = float(st.session_state.get("phase4q1_entry_key", 0.0) or 0.0)
            phase4q1_initial_shares = float(st.session_state.get("phase4q1_initial_shares_key", 0.0) or 0.0)
            phase4q1_remaining_shares = float(st.session_state.get("phase4q1_remaining_shares_key", 0.0) or 0.0)
            phase4q1_realized_pl = float(st.session_state.get("phase4q1_realized_pl_key", 0.0) or 0.0)
            phase4q1_initial_stop = float(st.session_state.get("phase4q1_initial_stop_key", 0.0) or 0.0)
            phase4q1_actual_stop = float(st.session_state.get("phase4q1_actual_stop_key", 0.0) or 0.0)
            phase4q3_test_mode = False
            phase4q3_test_mark = 0.0

            st.caption("Candidate mode: live-position entry, shares, P/L, stop, and simulation inputs are hidden.")

        else:
            st.markdown("### 📦 Close & Archive Trade")
            st.caption("Legitimate completed trades only. Final closeout fields are separate from live-entry fields.")
            # Phase 4Q.9C: Closed Trade must not depend on widget-bound values that
            # Streamlit discards when the live-position inputs are no longer rendered.
            # Re-read the durable live row every run and use it as the immutable
            # closeout source. This keeps the actual entry/share data available after
            # the user changes Position State from Entered / Live Position to Closed Trade.
            phase4q3_test_mode = False
            phase4q3_test_mark = 0.0
            phase4q9_source_row = None

            if phase4q1_ticker and _phase4q5_storage_config()["configured"]:
                try:
                    phase4q9_source_row = _phase4q5_load_latest_position(phase4q1_ticker)
                except Exception:
                    phase4q9_source_row = None

            if phase4q9_source_row:
                phase4q1_entry = float(phase4q9_source_row.get("entry") or 0.0)
                phase4q1_initial_shares = float(phase4q9_source_row.get("initial_shares") or 0.0)
                phase4q1_remaining_shares = float(phase4q9_source_row.get("remaining_shares") or 0.0)
                phase4q1_realized_pl = float(phase4q9_source_row.get("realized_pl") or 0.0)
                phase4q1_initial_stop = float(phase4q9_source_row.get("original_stop") or 0.0)
                phase4q1_actual_stop = float(phase4q9_source_row.get("current_stop_input") or 0.0)

                st.caption("Phase 4Q.9 closeout source — durable live-position record")
                close_src_1, close_src_2 = st.columns(2)
                close_src_1.metric("Actual entry", f"${phase4q1_entry:,.2f}")
                close_src_2.metric("Initial shares", f"{phase4q1_initial_shares:.5f}")
                close_src_3, close_src_4 = st.columns(2)
                close_src_3.metric("Saved remaining", f"{phase4q1_remaining_shares:.5f}")
                close_src_4.metric("Realized P/L saved", f"${phase4q1_realized_pl:,.2f}")
                st.caption(
                    "These values come directly from the durable live-position record. "
                    "Final exit price and final realized P/L are entered in Close & Archive below."
                )
            else:
                phase4q1_entry = 0.0
                phase4q1_initial_shares = 0.0
                phase4q1_remaining_shares = 0.0
                phase4q1_realized_pl = 0.0
                phase4q1_initial_stop = 0.0
                phase4q1_actual_stop = 0.0
                st.warning(
                    "No durable live-position source was found for this ticker. "
                    "Load the trade from Held Positions before selecting Closed Trade."
                )

        st.divider()
        st.caption("Saved / historical position tools")

        phase4q5_cfg = _phase4q5_storage_config()
        if phase4q5_cfg["configured"]:
            st.caption("Phase 4Q.5 durable storage: ✅ configured")

            st.markdown("**📌 Held Positions**")
            try:
                phase4q6_held = _phase4q6_list_held_positions()
                if phase4q6_held:
                    phase4q6_dashboard = _phase4q6_enrich_held_positions(phase4q6_held)

                    for row in phase4q6_dashboard:
                        ticker = str(row.get("ticker", "")).upper().strip()
                        entry = float(row.get("entry") or 0.0)
                        remaining = float(row.get("remaining_shares") or 0.0)
                        mark = float(row.get("mark") or 0.0)
                        unrealized = row.get("unrealized_pl", np.nan)
                        current_r = row.get("current_r", np.nan)
                        protective = float(row.get("protective_floor") or 0.0)
                        attention = str(row.get("attention") or "")

                        st.button(
                            ticker,
                            key=f"phase4q6_held_{ticker}",
                            on_click=_phase4q6_load_held_ticker_callback,
                            args=(ticker,),
                            use_container_width=True,
                            help=f"Load {ticker} into Position-State Manager",
                        )

                    st.caption(
                        f"{len(phase4q6_dashboard)} open swing position"
                        f"{'s' if len(phase4q6_dashboard) != 1 else ''} in durable storage."
                    )
                else:
                    st.caption("No open durable swing positions saved yet.")
            except Exception as exc:
                st.caption(f"Held-position list unavailable: {exc}")

        else:
            st.caption("Phase 4Q.5 durable storage: ⚠️ not configured")

        if phase4q5_cfg["configured"]:
            st.markdown("**👀 Saved Candidates**")
            try:
                phase4q8_candidates = _phase4q8_list_candidates()
                if phase4q8_candidates:
                    candidate_lookup = {
                        str(r.get("ticker", "")).upper().strip(): r
                        for r in phase4q8_candidates
                        if str(r.get("ticker", "")).strip()
                    }
                    candidate_options = [""] + sorted(candidate_lookup.keys())

                    selected_candidate = st.selectbox(
                        "Select saved candidate",
                        candidate_options,
                        key="phase4q8_selected_candidate",
                        format_func=lambda x: "— choose candidate —" if not x else (
                            f"{x} — {candidate_lookup.get(x, {}).get('candidate_action', '')}"
                        ),
                    )

                    st.button(
                        "🚀 Promote to Live Position",
                        on_click=_phase4q8_promote_candidate_callback,
                        disabled=not bool(selected_candidate),
                        use_container_width=True,
                    )
                    c_load, c_delete = st.columns(2)
                    with c_load:
                        st.button(
                            "Load",
                            on_click=_phase4q8_load_candidate_callback,
                            disabled=not bool(selected_candidate),
                            use_container_width=True,
                        )
                    with c_delete:
                        st.button(
                            "Delete",
                            on_click=_phase4q8_delete_candidate_callback,
                            disabled=not bool(selected_candidate),
                            use_container_width=True,
                        )

                    st.caption(
                        f"{len(candidate_lookup)} saved candidate"
                        f"{'s' if len(candidate_lookup) != 1 else ''}."
                    )
                else:
                    st.caption("No saved candidates yet.")
            except Exception as exc:
                st.caption(f"Saved Candidates unavailable: {exc}")

            if st.session_state.get("phase4q8_message"):
                st.caption(st.session_state["phase4q8_message"])

        st.markdown("#### 🗄️ Archive / Closed Trades Review")
        history_storage_cfg = _phase4q5_storage_config()
        try:
            closed_sidebar_rows = _phase4q9_load_closed_trades(100) if history_storage_cfg["configured"] else []
        except Exception as exc:
            closed_sidebar_rows = []
            st.caption(f"Closed Trades History unavailable: {exc}")

        if closed_sidebar_rows:
            labels = ["— choose closed trade —"]
            lookup = {}
            for r in closed_sidebar_rows:
                t = str(r.get("ticker") or "").upper()
                d = str(r.get("closed_at") or "")[:10]
                label = f"{t} — {d}" if d else t
                base, n = label, 2
                while label in lookup:
                    label = f"{base} #{n}"; n += 1
                labels.append(label); lookup[label] = r
            chosen = st.selectbox("Select closed trade", labels, key="phase4q9_history_selected")
            if chosen != "— choose closed trade —":
                r=lookup[chosen]
                e=float(r.get("entry") or 0); x=float(r.get("final_exit_price") or 0)
                p=float(r.get("realized_pl") or 0); s=float(r.get("initial_shares") or 0)
                st.caption(f"{str(r.get('ticker') or '').upper()} | Entry ${e:,.2f} → Exit ${x:,.2f} | {s:.5f} shares")
                st.caption(f"Realized P/L: {'+' if p>0 else ''}${p:,.2f} | {str(r.get('exit_reason') or '—')}")
                if r.get("highest_r") is not None:
                    st.caption(f"Highest R: {float(r.get('highest_r')):.2f}R | Highest state: {str(r.get('highest_state') or '—')}")
                if r.get("notes"):
                    st.caption(f"Notes: {r.get('notes')}")
        else:
            st.caption("No durable closed trades yet.")

        if st.session_state.get("phase4q5_last_message"):
            st.caption(st.session_state["phase4q5_last_message"])
        if st.session_state.get("phase4q9_message"):
            st.caption(st.session_state["phase4q9_message"])

if run_phase4q1:
    st.session_state["phase4q1_view_active"] = True

st.info(
    "Phase 4R.4B keeps Bullseye 4.0 / Phase 4Q management math frozen, preserves 4R.3/4R.4 logic, and adds cached market-cap context for information only. "
    "Close & Archive writes the completed trade to bullseye_closed_trades and removes the matching live row atomically, so a trade cannot remain half-closed in Held Positions. "
    "Delete Live Position remains reserved for erroneous/test records."
)

phase4q6_main_cfg = _phase4q5_storage_config()
if phase4q6_main_cfg["configured"]:
    try:
        phase4q6_main_rows = _phase4q6_list_held_positions()
        if phase4q6_main_rows:
            phase4q6_main = _phase4q6_enrich_held_positions(phase4q6_main_rows)
            st.subheader("📊 Phase 4Q.6B Held Positions Dashboard")

            dash_df = pd.DataFrame([
                {
                    "Ticker": str(r.get("ticker", "")).upper().strip(),
                    "Entry": float(r.get("entry") or 0.0),
                    "Mark": float(r.get("mark") or 0.0),
                    "Remaining": float(r.get("remaining_shares") or 0.0),
                    "Unrealized P/L": r.get("unrealized_pl", np.nan),
                    "Current R": r.get("current_r", np.nan),
                    "Highest State": r.get("highest_state") or "",
                    "Protective Stop": float(r.get("protective_floor") or 0.0),
                    "Attention": r.get("attention") or "",
                }
                for r in phase4q6_main
            ])

            st.dataframe(
                dash_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Entry": st.column_config.NumberColumn(format="$%.2f"),
                    "Mark": st.column_config.NumberColumn(format="$%.2f"),
                    "Remaining": st.column_config.NumberColumn(format="%.5f"),
                    "Unrealized P/L": st.column_config.NumberColumn(format="$%.2f"),
                    "Current R": st.column_config.NumberColumn(format="%.2fR"),
                    "Protective Stop": st.column_config.NumberColumn(format="$%.2f"),
                },
            )
    except Exception as exc:
        st.caption(f"Phase 4Q.6B dashboard unavailable: {exc}")

if run:
    with st.spinner("Downloading market data and scoring candidates..."):
        tickers2 = sorted(set(tickers + ["SPY"]))
        data = download_prices(tickers2)
        market_caps = get_market_caps(tuple(tickers))
        spy = one_symbol(data, "SPY")
        rows = []

        if spy is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                try:
                    row = score_stock(df, spy)
                    row["Ticker"] = t
                    row["Market Cap ($)"] = market_caps.get(t, np.nan)
                    row["Market Cap"] = _format_market_cap(row["Market Cap ($)"])
                    row.update(build_phase4r_early_warning(row))
                    plan_4r3 = build_trade_plan(df, row)
                    row.update(
                        build_phase4r3_opportunity_state(
                            row,
                            plan_4r3,
                            float(row.get("Price", np.nan)),
                        )
                    )
                    rows.append(row)
                except Exception:
                    continue

        if rows:
            result = pd.DataFrame(rows).sort_values(
                ["4R Stage Rank", "4R.3 Opportunity Rank", "4I Action Rank", "Bullseye 4.0 Score", "4R Readiness", "4H Core Count", "4.0 Accelerator"],
                ascending=[False, False, False, False, False, False, False],
            )

            # Phase 4R.2: persist this complete scanner run before displaying it.
            try:
                snapshot_result = _phase4r2_save_snapshot(result)
                try:
                    phase4r5_updated = _phase4r5_update_from_scan(result)
                except Exception:
                    phase4r5_updated = 0
                st.success(
                    f"Phase 4R.2 snapshot saved: {snapshot_result['saved']} tickers recorded for this scanner run."
                )
            except Exception as exc:
                st.warning(
                    "Phase 4R.2 snapshot was not saved. "
                    f"Run the Phase 4R.2 Supabase SQL first if this is the initial install. Details: {exc}"
                )

            st.subheader("🚦 Phase 4R.2 Developing Setup / Early Warning")
            st.caption(
                "QUALIFIED keeps Bullseye's existing 90+ threshold unchanged. "
                "DEVELOPING and WATCH are pre-qualification visibility only; they are not buy signals."
            )
            early = result[result["4R Stage"].isin(["DEVELOPING", "WATCH"])].copy()
            if not early.empty:
                st.dataframe(
                    early[
                        [
                            "Ticker", "Market Cap", "4R.3 Opportunity Signal", "4R Stage", "Bullseye 4.0 Score", "4R Gap to 90",
                            "4R Readiness", "4R Why", "4R Next Trigger",
                            "4.0 Accelerator", "4H Core Count", "Momentum Accel",
                            "Rel Vol", "RS vs SPY 20D", "RSI", "Dist 20MA %", "Price",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No WATCH or DEVELOPING setups are present in this scan.")

            stage_counts = result["4R Stage"].value_counts()
            r1, r2, r3 = st.columns(3)
            r1.metric("Qualified", int(stage_counts.get("QUALIFIED", 0)))
            r2.metric("Developing", int(stage_counts.get("DEVELOPING", 0)))
            r3.metric("Watch", int(stage_counts.get("WATCH", 0)))

            st.markdown("#### 🔎 Investigate a scanner result")
            phase4r2a_pool = result[
                result["4R Stage"].isin(["QUALIFIED", "DEVELOPING", "WATCH"])
            ].copy()
            if phase4r2a_pool.empty:
                st.caption("No WATCH, DEVELOPING, or QUALIFIED ticker is available from this scan.")
            else:
                phase4r2a_options = phase4r2a_pool["Ticker"].astype(str).tolist()
                phase4r2a_lookup = phase4r2a_pool.set_index(
                    phase4r2a_pool["Ticker"].astype(str)
                )
                phase4r2a_ticker = st.selectbox(
                    "Ticker from this scan",
                    phase4r2a_options,
                    key="phase4r2a_scanner_ticker",
                    format_func=lambda t: (
                        f"{t} — {phase4r2a_lookup.loc[str(t), '4R.3 Opportunity Signal']} — "
                        f"{phase4r2a_lookup.loc[str(t), '4R Stage']} — "
                        f"{float(phase4r2a_lookup.loc[str(t), 'Bullseye 4.0 Score']):.1f}"
                    ),
                )
                if st.button(
                    f"🔎 Investigate {phase4r2a_ticker}",
                    key="phase4r2a_investigate_button",
                ):
                    # Defer widget-bound mutations until the next rerun.
                    st.session_state["phase4r2a_investigate_ticker"] = str(phase4r2a_ticker)
                    st.rerun()

            st.subheader("🏆 Top Bullseye Opportunities")
            st.dataframe(
                result[
                    [
                        "Ticker", "Market Cap", "4R.3 Opportunity Signal", "4R Stage", "4I Action", "4H Signal Tier", "Bullseye 4.0 Score",
                        "4R Gap to 90", "4R Readiness",
                        "4H Signal Badges", "4I Why", "Price",
                        "4H Core Count", "4.0 Accelerator", "Beta 120D",
                        "Avg $ Volume 60D ($M)", "120D %",
                        "5D %", "20D %", "60D %", "RSI", "Dist 20MA %",
                        "Rel Vol", "RS vs SPY 20D", "Market Regime",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download results CSV",
                result.to_csv(index=False),
                "bullseye_phase4r2a_results.csv",
                "text/csv",
            )
        else:
            st.warning("No usable candidates were returned.")


if run_phase4r2_history:
    st.divider()
    st.subheader(f"🕒 Phase 4R.2 Snapshot History — {phase4r2_history_ticker or 'Ticker'}")
    st.caption(
        "Diagnostic history only. Each row is what Bullseye knew at one scanner run; "
        "no historical row changes Bullseye 4.0 scoring."
    )
    try:
        history_records = _phase4r2_load_history(phase4r2_history_ticker, limit=250)
        history_df = _phase4r2_history_frame(history_records)

        if history_df.empty:
            st.info(
                f"No Phase 4R.2 snapshots are stored yet for {phase4r2_history_ticker}. "
                "Run the scanner with that ticker in the universe to begin its history."
            )
        else:
            display_cols = [
                "Scanned ET", "Opportunity Icon", "Opportunity State", "Stage", "Score", "Gap to 90", "Readiness", "Price",
                "Accelerator", "Core Count", "Momentum Accel", "Rel Vol",
                "RS vs SPY 20D", "RSI", "Dist 20MA %", "4H Tier", "4I Action",
                "Market Regime", "Opportunity Why", "Opportunity Next Trigger", "Why", "Next Trigger",
            ]
            display_cols = [c for c in display_cols if c in history_df.columns]
            st.dataframe(
                history_df[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Score": st.column_config.NumberColumn(format="%.1f"),
                    "Gap to 90": st.column_config.NumberColumn(format="%.1f"),
                    "Accelerator": st.column_config.NumberColumn(format="%.1f"),
                    "Momentum Accel": st.column_config.NumberColumn(format="%.2f"),
                    "Rel Vol": st.column_config.NumberColumn(format="%.2f"),
                    "RS vs SPY 20D": st.column_config.NumberColumn(format="%.2f"),
                    "RSI": st.column_config.NumberColumn(format="%.1f"),
                    "Dist 20MA %": st.column_config.NumberColumn(format="%.2f%%"),
                },
            )

            first = history_df.iloc[0]
            latest = history_df.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Snapshots", len(history_df))
            c2.metric(
                "First → Latest Score",
                f"{float(latest.get('Score', np.nan)):.1f}",
                f"{float(latest.get('Score', np.nan)) - float(first.get('Score', np.nan)):+.1f}",
            )
            c3.metric(
                "First → Latest Price",
                f"${float(latest.get('Price', np.nan)):.2f}",
                f"{float(latest.get('Price', np.nan)) - float(first.get('Price', np.nan)):+.2f}",
            )
            latest_opp_icon = str(latest.get("Opportunity Icon", ""))
            latest_opp_state = str(latest.get("Opportunity State", "—"))
            c4.metric("Latest Opportunity", f"{latest_opp_icon} {latest_opp_state}".strip())

            st.download_button(
                f"Download {phase4r2_history_ticker} snapshot history CSV",
                history_df.to_csv(index=False),
                f"bullseye_4r2_{phase4r2_history_ticker.lower()}_snapshot_history.csv",
                "text/csv",
            )
    except Exception as exc:
        st.error(f"Phase 4R.2 snapshot history unavailable: {exc}")


if run_phase4r5_outcomes:
    st.divider()
    st.subheader("🎯 Phase 4R.5 Candidate Outcome Journal")
    st.caption("Observed scanner outcomes only. A check means Bullseye observed the event on a scanner run; it is not proof of an intraday touch between scans and it does not change any scoring or management rule.")
    try:
        recs=_phase4r5_list_outcomes(phase4r5_outcome_ticker or None)
        odf=_phase4r5_frame(recs)
        if odf.empty:
            st.info("No tracked candidate outcomes yet. Save a ticker to Candidate Watchlist to begin tracking it.")
        else:
            st.dataframe(odf, use_container_width=True, hide_index=True,
                column_config={"Initial Score":st.column_config.NumberColumn(format="%.1f"),"Latest Score":st.column_config.NumberColumn(format="%.1f"),
                "Initial Price":st.column_config.NumberColumn(format="$%.2f"),"Latest Price":st.column_config.NumberColumn(format="$%.2f"),
                "Max Observed":st.column_config.NumberColumn(format="$%.2f"),"Min Observed":st.column_config.NumberColumn(format="$%.2f")})
            st.download_button("Download 4R.5 candidate outcomes CSV", odf.to_csv(index=False), "bullseye_4r5_candidate_outcomes.csv", "text/csv")
    except Exception as exc:
        st.error(f"Phase 4R.5 Candidate Outcome Journal unavailable: {exc}")


if run_backtest:
    with st.spinner("Running historical Bullseye validation..."):
        tickers2 = sorted(set(tickers + ["SPY"]))
        data = download_prices(tickers2)
        spy = one_symbol(data, "SPY")
        bt_rows = []

        if spy is None:
            st.error("Could not retrieve SPY data for backtesting.")
        else:
            for t in tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                bt_rows.extend(
                    backtest_symbol(
                        df,
                        spy,
                        t,
                        lookback_days=backtest_lookback,
                        step=backtest_step,
                    )
                )

        if bt_rows:
            bt = pd.DataFrame(bt_rows)

            st.subheader("🧪 Bullseye Historical Validation")
            st.caption(
                "Each row is a historical Bullseye snapshot using only information available on that date. "
                "Forward returns show what happened afterward."
            )

            buckets = pd.cut(
                bt["Opportunity Score"],
                bins=[-0.01, 49.99, 64.99, 74.99, 84.99, 100],
                labels=["<50", "50–64.9", "65–74.9", "75–84.9", "85+"],
            )
            bt["Score Bucket"] = buckets

            summary = (
                bt.groupby("Score Bucket", observed=True)
                .agg(
                    Samples=("Ticker", "count"),
                    Avg_5D=("5D Forward %", "mean"),
                    Avg_10D=("10D Forward %", "mean"),
                    Avg_15D=("15D Forward %", "mean"),
                    Avg_20D=("20D Forward %", "mean"),
                    Win_5D=("5D Forward %", lambda x: (x > 0).mean() * 100),
                    Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                    Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                )
                .reset_index()
            )

            for col in ["Avg_5D", "Avg_10D", "Avg_15D", "Avg_20D", "Win_5D", "Win_20D", "Hit_5pct_20D"]:
                summary[col] = summary[col].round(2)

            st.markdown("**Results by Opportunity Score bucket**")
            st.dataframe(summary, use_container_width=True, hide_index=True)

            # Compare stronger Opportunity Scores with the full sample.
            q75 = bt["Opportunity Score"].quantile(0.75)
            q90 = bt["Opportunity Score"].quantile(0.90)
            comparison_rows = []
            for name, subset in [
                ("All samples", bt),
                ("Top 25% Opportunity", bt[bt["Opportunity Score"] >= q75]),
                ("Top 10% Opportunity", bt[bt["Opportunity Score"] >= q90]),
            ]:
                comparison_rows.append({
                    "Group": name,
                    "Samples": len(subset),
                    "Avg 5D %": round(subset["5D Forward %"].mean(), 2),
                    "Avg 10D %": round(subset["10D Forward %"].mean(), 2),
                    "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                    "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                    "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                })

            st.markdown("**Opportunity-score percentile comparison**")
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

            comparison_score_rows = []
            for score_name in ["Bullseye Score", "Opportunity Score", "Experimental 3.0 Score"]:
                temp = bt[[score_name, "5D Forward %", "10D Forward %", "15D Forward %", "20D Forward %"]].dropna()
                if len(temp) < 50:
                    continue

                q80 = temp[score_name].quantile(0.80)
                q90 = temp[score_name].quantile(0.90)

                for group_name, subset in [
                    ("All", temp),
                    ("Top 20%", temp[temp[score_name] >= q80]),
                    ("Top 10%", temp[temp[score_name] >= q90]),
                ]:
                    comparison_score_rows.append({
                        "Score System": score_name,
                        "Group": group_name,
                        "Samples": len(subset),
                        "Avg 5D %": round(subset["5D Forward %"].mean(), 2),
                        "Avg 10D %": round(subset["10D Forward %"].mean(), 2),
                        "Avg 15D %": round(subset["15D Forward %"].mean(), 2),
                        "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                        "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                        "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                    })

            st.markdown("**Phase 3C score-system head-to-head**")
            st.dataframe(
                pd.DataFrame(comparison_score_rows),
                use_container_width=True,
                hide_index=True,
            )

            exp_bt = bt.copy()
            exp_bt["Experimental 3.0 Bucket"] = pd.cut(
                exp_bt["Experimental 3.0 Score"],
                bins=[-0.01, 59.99, 69.99, 79.99, 89.99, 100],
                labels=["<60", "60–69.9", "70–79.9", "80–89.9", "90+"],
            )
            exp_summary = (
                exp_bt.groupby("Experimental 3.0 Bucket", observed=True)
                .agg(
                    Samples=("Ticker", "count"),
                    Avg_5D=("5D Forward %", "mean"),
                    Avg_10D=("10D Forward %", "mean"),
                    Avg_15D=("15D Forward %", "mean"),
                    Avg_20D=("20D Forward %", "mean"),
                    Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                    Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                )
                .reset_index()
            )
            for col in ["Avg_5D", "Avg_10D", "Avg_15D", "Avg_20D", "Win_20D", "Hit_5pct_20D"]:
                exp_summary[col] = exp_summary[col].round(2)

            st.markdown("**Experimental 3.0 score buckets**")
            st.dataframe(exp_summary, use_container_width=True, hide_index=True)

            st.subheader("🔬 Bullseye Signal Lab")
            st.caption(
                "Signal Lab measures historical relationships without changing the live scanner yet."
            )

            signal_cols = [
                "Bullseye Score", "Opportunity Score", "Experimental 3.0 Score", "Momentum", "Volume",
                "Relative Strength", "Technical", "Setup Quality",
                "Extension Penalty", "RSI", "Dist 20MA %", "Momentum Accel",
                "Market Regime", "Risk/Liquidity",
            ]
            forward_cols = ["5D Forward %", "10D Forward %", "15D Forward %", "20D Forward %"]

            corr_rows = []
            for signal in signal_cols:
                row = {"Signal": signal}
                for fwd in forward_cols:
                    pair = bt[[signal, fwd]].dropna()
                    row[f"Pearson {fwd.split()[0]}"] = round(
                        pair[signal].corr(pair[fwd], method="pearson"), 3
                    ) if len(pair) >= 10 else np.nan
                    if len(pair) >= 10:
                        ranked_signal = pair[signal].rank(method="average")
                        ranked_fwd = pair[fwd].rank(method="average")
                        row[f"Spearman {fwd.split()[0]}"] = round(
                            ranked_signal.corr(ranked_fwd), 3
                        )
                    else:
                        row[f"Spearman {fwd.split()[0]}"] = np.nan
                corr_rows.append(row)

            st.markdown("**Signal correlation by forward horizon**")
            st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)

            quintile_rows = []
            for signal in signal_cols:
                temp = bt[[signal] + forward_cols].dropna().copy()
                if len(temp) < 50 or temp[signal].nunique() < 3:
                    continue
                try:
                    temp["Q"] = pd.qcut(temp[signal], q=5, duplicates="drop")
                except Exception:
                    continue
                grp = temp.groupby("Q", observed=True).agg(
                    Samples=(signal, "count"),
                    Signal_Min=(signal, "min"),
                    Signal_Max=(signal, "max"),
                    Avg_5D=("5D Forward %", "mean"),
                    Avg_10D=("10D Forward %", "mean"),
                    Avg_20D=("20D Forward %", "mean"),
                    Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                    Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                ).reset_index(drop=True)
                if len(grp) >= 2:
                    low, high = grp.iloc[0], grp.iloc[-1]
                    quintile_rows.append({
                        "Signal": signal,
                        "Low 20% Avg 20D": round(low["Avg_20D"], 2),
                        "High 20% Avg 20D": round(high["Avg_20D"], 2),
                        "High-Low Spread": round(high["Avg_20D"] - low["Avg_20D"], 2),
                        "Low 20% Win %": round(low["Win_20D"], 2),
                        "High 20% Win %": round(high["Win_20D"], 2),
                        "High 20% Hit 5%": round(high["Hit_5pct_20D"], 2),
                    })

            if quintile_rows:
                st.markdown("**Signal quintile spread — highest 20% vs lowest 20%**")
                qsum = pd.DataFrame(quintile_rows).sort_values("High-Low Spread", ascending=False)
                st.dataframe(qsum, use_container_width=True, hide_index=True)

            rsi_study = bt.copy()
            rsi_study["RSI Range"] = pd.cut(
                rsi_study["RSI"],
                bins=[0, 40, 50, 60, 70, 80, 100],
                labels=["<40", "40–49.9", "50–59.9", "60–69.9", "70–79.9", "80+"],
                include_lowest=True,
            )
            rsi_summary = rsi_study.groupby("RSI Range", observed=True).agg(
                Samples=("Ticker", "count"),
                Avg_5D=("5D Forward %", "mean"),
                Avg_10D=("10D Forward %", "mean"),
                Avg_20D=("20D Forward %", "mean"),
                Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
            ).reset_index()

            dist_study = bt.copy()
            dist_study["20MA Distance Range"] = pd.cut(
                dist_study["Dist 20MA %"],
                bins=[-100, -5, 0, 5, 10, 15, 20, 1000],
                labels=["<-5%", "-5–0%", "0–5%", "5–10%", "10–15%", "15–20%", "20%+"],
                include_lowest=True,
            )
            dist_summary = dist_study.groupby("20MA Distance Range", observed=True).agg(
                Samples=("Ticker", "count"),
                Avg_5D=("5D Forward %", "mean"),
                Avg_10D=("10D Forward %", "mean"),
                Avg_20D=("20D Forward %", "mean"),
                Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
            ).reset_index()

            for frame in (rsi_summary, dist_summary):
                for col in frame.columns:
                    if col not in ("RSI Range", "20MA Distance Range", "Samples"):
                        frame[col] = frame[col].round(2)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**RSI range performance**")
                st.dataframe(rsi_summary, use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**Distance from 20-day MA performance**")
                st.dataframe(dist_summary, use_container_width=True, hide_index=True)

            st.markdown("**Historical snapshots**")
            st.dataframe(
                bt.sort_values(["Date", "Opportunity Score"], ascending=[False, False]),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download backtest CSV",
                bt.to_csv(index=False),
                "bullseye_phase3c_signal_lab.csv",
                "text/csv",
            )
        else:
            st.warning("No historical backtest samples were returned.")


if run_multi_period:
    with st.spinner("Running multi-period Bullseye validation..."):
        tickers2 = sorted(set(tickers + ["SPY"]))
        data = download_prices(tickers2)

        periods = [
            ("1Y", 252),
            ("2Y", 504),
            ("3Y", 756),
            ("5Y", 1260),
        ]

        period_df = run_period_validation(
            data,
            tickers,
            periods=periods,
            step=backtest_step,
        )

        if len(period_df):
            st.subheader("📚 Multi-Period Signal Validation")
            st.caption(
                "Positive High-Low Spread means the highest 20% of that signal outperformed "
                "the lowest 20% over the following 20 trading days."
            )

            st.markdown("**All signal-period results**")
            st.dataframe(
                period_df.sort_values(["Signal", "Period"]),
                use_container_width=True,
                hide_index=True,
            )

            consistency = (
                period_df.groupby("Signal", observed=True)
                .agg(
                    Periods_Tested=("Period", "nunique"),
                    Avg_Spread=("High-Low Spread", "mean"),
                    Median_Spread=("High-Low Spread", "median"),
                    Positive_Periods=("High-Low Spread", lambda x: (x > 0).sum()),
                    Best_Spread=("High-Low Spread", "max"),
                    Worst_Spread=("High-Low Spread", "min"),
                    Avg_High20_Win=("High 20% Win %", "mean"),
                    Avg_High20_Hit5=("High 20% Hit 5%", "mean"),
                )
                .reset_index()
            )

            consistency["Consistency %"] = (
                consistency["Positive_Periods"] / consistency["Periods_Tested"] * 100
            ).round(1)

            for col in [
                "Avg_Spread", "Median_Spread", "Best_Spread", "Worst_Spread",
                "Avg_High20_Win", "Avg_High20_Hit5"
            ]:
                consistency[col] = consistency[col].round(2)

            consistency = consistency.sort_values(
                ["Consistency %", "Avg_Spread"],
                ascending=[False, False],
            )

            st.markdown("**Signal consistency across periods**")
            st.dataframe(
                consistency,
                use_container_width=True,
                hide_index=True,
            )

            # Focused table for the signals that challenged our earlier assumptions.
            focus_signals = [
                "Relative Strength",
                "RSI",
                "Dist 20MA %",
                "Extension Penalty",
                "Setup Quality",
                "Risk/Liquidity",
                "Market Regime",
            ]
            focus = period_df[period_df["Signal"].isin(focus_signals)].copy()

            st.markdown("**Key signals by period**")
            st.dataframe(
                focus.sort_values(["Signal", "Period"]),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download multi-period validation CSV",
                period_df.to_csv(index=False),
                "bullseye_phase3c_multi_period.csv",
                "text/csv",
            )
        else:
            st.warning("No multi-period validation results were returned.")


if run_walk_forward:
    with st.spinner("Running frozen 3.0 walk-forward validation..."):
        tickers2 = sorted(set(tickers + ["SPY"]))
        data = download_prices(tickers2)
        spy = one_symbol(data, "SPY")
        wf_rows = []

        if spy is None:
            st.error("Could not retrieve SPY data.")
        else:
            # Build one 5-year snapshot set, then divide it chronologically.
            all_rows = []
            for t in tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                all_rows.extend(backtest_symbol(df, spy, t, lookback_days=1260, step=backtest_step))

            if all_rows:
                wf = pd.DataFrame(all_rows).dropna(subset=["Date"]).copy()
                wf["Date"] = pd.to_datetime(wf["Date"])
                unique_dates = sorted(wf["Date"].unique())

                if len(unique_dates) >= 3:
                    # Three non-overlapping chronological blocks with approximately equal trading dates.
                    cuts = np.array_split(np.array(unique_dates), 3)
                    period_names = ["Older period", "Middle period", "Recent period"]

                    score_systems = ["Bullseye Score", "Opportunity Score", "Experimental 3.0 Score"]

                    for period_name, dates in zip(period_names, cuts):
                        if len(dates) == 0:
                            continue
                        start_date = pd.Timestamp(dates[0])
                        end_date = pd.Timestamp(dates[-1])
                        period_bt = wf[(wf["Date"] >= start_date) & (wf["Date"] <= end_date)].copy()

                        for score_name in score_systems:
                            temp = period_bt[[score_name, "5D Forward %", "10D Forward %",
                                              "15D Forward %", "20D Forward %"]].dropna()
                            if len(temp) < 30:
                                continue

                            q80 = temp[score_name].quantile(0.80)
                            q90 = temp[score_name].quantile(0.90)

                            for group_name, subset in [
                                ("Top 20%", temp[temp[score_name] >= q80]),
                                ("Top 10%", temp[temp[score_name] >= q90]),
                            ]:
                                if len(subset) == 0:
                                    continue
                                wf_rows.append({
                                    "Period": period_name,
                                    "Start": start_date.date(),
                                    "End": end_date.date(),
                                    "Score System": score_name,
                                    "Group": group_name,
                                    "Samples": len(subset),
                                    "Avg 5D %": round(subset["5D Forward %"].mean(), 2),
                                    "Avg 10D %": round(subset["10D Forward %"].mean(), 2),
                                    "Avg 15D %": round(subset["15D Forward %"].mean(), 2),
                                    "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                                    "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                                    "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                                })

                    wf_result = pd.DataFrame(wf_rows)
                    if len(wf_result):
                        st.subheader("🚶 Phase 3D Walk-Forward Validation")
                        st.caption(
                            "Experimental 3.0 is frozen exactly as built in Phase 3C. "
                            "These are separate chronological blocks, not overlapping 1Y/2Y/3Y/5Y windows."
                        )
                        st.markdown("**Head-to-head by independent time block**")
                        st.dataframe(wf_result, use_container_width=True, hide_index=True)

                        top10 = wf_result[wf_result["Group"] == "Top 10%"].copy()
                        summary = (
                            top10.groupby("Score System", observed=True)
                            .agg(
                                Periods=("Period", "nunique"),
                                Avg_20D=("Avg 20D %", "mean"),
                                Worst_Period_20D=("Avg 20D %", "min"),
                                Best_Period_20D=("Avg 20D %", "max"),
                                Avg_Win_20D=("20D Win %", "mean"),
                                Avg_Hit5_20D=("20D Hit 5% %", "mean"),
                            )
                            .reset_index()
                        )
                        for col in ["Avg_20D", "Worst_Period_20D", "Best_Period_20D",
                                    "Avg_Win_20D", "Avg_Hit5_20D"]:
                            summary[col] = summary[col].round(2)

                        st.markdown("**Top-10% walk-forward summary**")
                        st.dataframe(
                            summary.sort_values("Avg_20D", ascending=False),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.download_button(
                            "Download Phase 3D walk-forward CSV",
                            wf_result.to_csv(index=False),
                            "bullseye_phase3d_walk_forward.csv",
                            "text/csv",
                        )
                    else:
                        st.warning("No walk-forward results were returned.")
                else:
                    st.warning("Not enough historical dates for the walk-forward split.")
            else:
                st.warning("No historical snapshots were returned.")


if run_stress_test:
    with st.spinner("Running Phase 3E stress test..."):
        tickers2 = sorted(set(tickers + ["SPY"]))
        data = download_prices(tickers2)
        spy = one_symbol(data, "SPY")
        stress_rows = []

        if spy is None:
            st.error("Could not retrieve SPY data.")
        else:
            # Fixed 20-trading-day snapshots reduce overlap between observations.
            for t in tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                stress_rows.extend(
                    backtest_symbol(
                        df,
                        spy,
                        t,
                        lookback_days=1260,
                        step=20,
                    )
                )

        if stress_rows:
            stress = pd.DataFrame(stress_rows).copy()

            st.subheader("🧱 Phase 3E Stress Test")
            st.caption(
                "Experimental 3.0 remains frozen. This test uses snapshots every 20 trading days "
                "to reduce overlap, then checks whether results are broad across tickers and market regimes."
            )

            # ------------------------------------------------------------
            # 1) Head-to-head using lower-overlap monthly snapshots
            # ------------------------------------------------------------
            score_rows = []
            for score_name in ["Bullseye Score", "Opportunity Score", "Experimental 3.0 Score"]:
                temp = stress[
                    [score_name, "5D Forward %", "10D Forward %", "15D Forward %", "20D Forward %"]
                ].dropna()

                if len(temp) < 50:
                    continue

                q80 = temp[score_name].quantile(0.80)
                q90 = temp[score_name].quantile(0.90)

                for group_name, subset in [
                    ("All", temp),
                    ("Top 20%", temp[temp[score_name] >= q80]),
                    ("Top 10%", temp[temp[score_name] >= q90]),
                ]:
                    if len(subset) == 0:
                        continue
                    score_rows.append({
                        "Score System": score_name,
                        "Group": group_name,
                        "Samples": len(subset),
                        "Avg 5D %": round(subset["5D Forward %"].mean(), 2),
                        "Avg 10D %": round(subset["10D Forward %"].mean(), 2),
                        "Avg 15D %": round(subset["15D Forward %"].mean(), 2),
                        "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                        "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                        "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                    })

            monthly_compare = pd.DataFrame(score_rows)
            st.markdown("**A. Lower-overlap score-system head-to-head**")
            st.dataframe(monthly_compare, use_container_width=True, hide_index=True)

            # ------------------------------------------------------------
            # 2) Ticker-by-ticker breadth for Experimental 3.0
            # ------------------------------------------------------------
            ticker_rows = []
            for ticker, grp in stress.groupby("Ticker", observed=True):
                grp = grp.dropna(subset=["Experimental 3.0 Score", "20D Forward %"]).copy()
                if len(grp) < 8:
                    continue

                q80 = grp["Experimental 3.0 Score"].quantile(0.80)
                top = grp[grp["Experimental 3.0 Score"] >= q80]
                if len(top) < 2:
                    continue

                ticker_rows.append({
                    "Ticker": ticker,
                    "Samples": len(grp),
                    "Top20 Samples": len(top),
                    "All Avg 20D %": round(grp["20D Forward %"].mean(), 2),
                    "Top20 Avg 20D %": round(top["20D Forward %"].mean(), 2),
                    "Top20 Excess %": round(
                        top["20D Forward %"].mean() - grp["20D Forward %"].mean(), 2
                    ),
                    "Top20 Win %": round((top["20D Forward %"] > 0).mean() * 100, 2),
                    "Top20 Hit 5% %": round((top["20D Forward %"] >= 5).mean() * 100, 2),
                })

            ticker_df = pd.DataFrame(ticker_rows)
            if len(ticker_df):
                ticker_df = ticker_df.sort_values("Top20 Excess %", ascending=False)
                positive_tickers = int((ticker_df["Top20 Excess %"] > 0).sum())
                tested_tickers = len(ticker_df)

                breadth_summary = pd.DataFrame([{
                    "Tickers Tested": tested_tickers,
                    "Tickers With Positive Top20 Excess": positive_tickers,
                    "Positive Breadth %": round(positive_tickers / tested_tickers * 100, 2)
                        if tested_tickers else np.nan,
                    "Median Top20 Excess %": round(ticker_df["Top20 Excess %"].median(), 2),
                    "Mean Top20 Excess %": round(ticker_df["Top20 Excess %"].mean(), 2),
                }])

                st.markdown("**B. Experimental 3.0 ticker-breadth summary**")
                st.dataframe(breadth_summary, use_container_width=True, hide_index=True)

                with st.expander("Ticker-by-ticker Experimental 3.0 results"):
                    st.dataframe(ticker_df, use_container_width=True, hide_index=True)

            # ------------------------------------------------------------
            # 3) Strong-vs-weak market regime
            # ------------------------------------------------------------
            regime = stress.dropna(
                subset=["Experimental 3.0 Score", "Market Regime", "20D Forward %"]
            ).copy()

            regime["Regime Group"] = np.where(
                regime["Market Regime"] >= 7,
                "Stronger market",
                "Weaker market",
            )

            regime_rows = []
            for regime_name, grp in regime.groupby("Regime Group", observed=True):
                if len(grp) < 20:
                    continue

                q80 = grp["Experimental 3.0 Score"].quantile(0.80)
                q90 = grp["Experimental 3.0 Score"].quantile(0.90)

                for group_name, subset in [
                    ("All", grp),
                    ("Top 20%", grp[grp["Experimental 3.0 Score"] >= q80]),
                    ("Top 10%", grp[grp["Experimental 3.0 Score"] >= q90]),
                ]:
                    if len(subset) == 0:
                        continue
                    regime_rows.append({
                        "Market Regime": regime_name,
                        "Group": group_name,
                        "Samples": len(subset),
                        "Avg 5D %": round(subset["5D Forward %"].mean(), 2),
                        "Avg 10D %": round(subset["10D Forward %"].mean(), 2),
                        "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                        "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                        "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                    })

            regime_df = pd.DataFrame(regime_rows)
            st.markdown("**C. Experimental 3.0 by market regime**")
            st.dataframe(regime_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download Phase 3E stress-test snapshots CSV",
                stress.to_csv(index=False),
                "bullseye_phase3e_stress_test.csv",
                "text/csv",
            )
        else:
            st.warning("No Phase 3E stress-test samples were returned.")


if run_diagnostics:
    with st.spinner("Running Phase 3F ticker-breadth diagnostics..."):
        tickers2 = sorted(set(tickers + ["SPY"]))
        data = download_prices(tickers2)
        spy = one_symbol(data, "SPY")
        diag_rows = []
        snapshot_rows = []

        if spy is None:
            st.error("Could not retrieve SPY data.")
        else:
            spy_returns = spy["Close"].pct_change().dropna()

            for t in tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue

                # Use 20-day snapshots, matching the Phase 3E breadth stress test.
                bt_rows = backtest_symbol(
                    df,
                    spy,
                    t,
                    lookback_days=1260,
                    step=20,
                )
                snapshot_rows.extend(bt_rows)

                c = df["Close"]
                v = df["Volume"]
                returns = c.pct_change().dropna()

                common = returns.index.intersection(spy_returns.index)
                beta = np.nan
                if len(common) >= 60:
                    stock_r = returns.loc[common]
                    spy_r = spy_returns.loc[common]
                    spy_var = float(spy_r.var())
                    if spy_var > 0:
                        beta = float(stock_r.cov(spy_r) / spy_var)

                ann_vol = float(returns.tail(252).std() * math.sqrt(252) * 100) if len(returns) >= 60 else np.nan
                avg_dollar_vol = float((c * v).tail(60).mean()) if len(c) >= 60 else np.nan
                ret_60 = pct(float(c.iloc[-1]), float(c.iloc[-61])) if len(c) >= 61 else np.nan
                ret_120 = pct(float(c.iloc[-1]), float(c.iloc[-121])) if len(c) >= 121 else np.nan
                ma20 = float(c.rolling(20).mean().iloc[-1])
                ma50 = float(c.rolling(50).mean().iloc[-1])
                dist20_now = pct(float(c.iloc[-1]), ma20)

                diag_rows.append({
                    "Ticker": t,
                    "Beta vs SPY": round(beta, 2) if pd.notna(beta) else np.nan,
                    "Ann Vol %": round(ann_vol, 2) if pd.notna(ann_vol) else np.nan,
                    "Avg $ Volume 60D ($M)": round(avg_dollar_vol / 1_000_000, 1)
                        if pd.notna(avg_dollar_vol) else np.nan,
                    "60D Return %": round(ret_60, 2) if pd.notna(ret_60) else np.nan,
                    "120D Return %": round(ret_120, 2) if pd.notna(ret_120) else np.nan,
                    "Current Dist 20MA %": round(dist20_now, 2) if pd.notna(dist20_now) else np.nan,
                    "Trend Above 20/50": bool(c.iloc[-1] > ma20 > ma50),
                })

        if snapshot_rows and diag_rows:
            snap = pd.DataFrame(snapshot_rows)
            characteristics = pd.DataFrame(diag_rows)

            # Recreate ticker-level Experimental 3.0 breadth outcome.
            breadth_rows = []
            for ticker, grp in snap.groupby("Ticker", observed=True):
                grp = grp.dropna(subset=["Experimental 3.0 Score", "20D Forward %"]).copy()
                if len(grp) < 8:
                    continue
                q80 = grp["Experimental 3.0 Score"].quantile(0.80)
                top = grp[grp["Experimental 3.0 Score"] >= q80]
                if len(top) < 2:
                    continue

                breadth_rows.append({
                    "Ticker": ticker,
                    "All Avg 20D %": grp["20D Forward %"].mean(),
                    "Top20 Avg 20D %": top["20D Forward %"].mean(),
                    "Top20 Excess %": top["20D Forward %"].mean() - grp["20D Forward %"].mean(),
                    "Top20 Win %": (top["20D Forward %"] > 0).mean() * 100,
                    "Top20 Hit 5% %": (top["20D Forward %"] >= 5).mean() * 100,
                    "Top20 Samples": len(top),
                })

            breadth = pd.DataFrame(breadth_rows)
            diag = breadth.merge(characteristics, on="Ticker", how="left")
            diag["3.0 Result"] = np.where(diag["Top20 Excess %"] > 0, "Winner", "Loser")

            for col in ["All Avg 20D %", "Top20 Avg 20D %", "Top20 Excess %",
                        "Top20 Win %", "Top20 Hit 5% %"]:
                diag[col] = diag[col].round(2)

            st.subheader("🧬 Phase 3F Breadth Diagnostics")
            st.caption(
                "This does not change Experimental 3.0. It compares tickers where the model added value "
                "with tickers where its top-ranked setups underperformed that ticker's own baseline."
            )

            # Winner-vs-loser characteristic averages.
            compare_cols = [
                "Beta vs SPY", "Ann Vol %", "Avg $ Volume 60D ($M)",
                "60D Return %", "120D Return %", "Current Dist 20MA %"
            ]

            group_compare = (
                diag.groupby("3.0 Result", observed=True)[compare_cols]
                .mean()
                .round(2)
                .reset_index()
            )

            st.markdown("**A. Winner vs loser characteristics**")
            st.dataframe(group_compare, use_container_width=True, hide_index=True)

            # Bucket diagnostics for volatility and beta.
            bucket_source = diag.dropna(subset=["Ann Vol %", "Beta vs SPY"]).copy()

            if len(bucket_source) >= 12:
                try:
                    bucket_source["Volatility Group"] = pd.qcut(
                        bucket_source["Ann Vol %"],
                        q=3,
                        labels=["Low vol", "Mid vol", "High vol"],
                        duplicates="drop",
                    )
                except Exception:
                    bucket_source["Volatility Group"] = "All"

                try:
                    bucket_source["Beta Group"] = pd.qcut(
                        bucket_source["Beta vs SPY"],
                        q=3,
                        labels=["Low beta", "Mid beta", "High beta"],
                        duplicates="drop",
                    )
                except Exception:
                    bucket_source["Beta Group"] = "All"

                vol_summary = (
                    bucket_source.groupby("Volatility Group", observed=True)
                    .agg(
                        Tickers=("Ticker", "count"),
                        Positive_Breadth=("Top20 Excess %", lambda x: (x > 0).mean() * 100),
                        Avg_Excess=("Top20 Excess %", "mean"),
                        Median_Excess=("Top20 Excess %", "median"),
                        Avg_Top20_Win=("Top20 Win %", "mean"),
                        Avg_Top20_Hit5=("Top20 Hit 5% %", "mean"),
                    )
                    .reset_index()
                )

                beta_summary = (
                    bucket_source.groupby("Beta Group", observed=True)
                    .agg(
                        Tickers=("Ticker", "count"),
                        Positive_Breadth=("Top20 Excess %", lambda x: (x > 0).mean() * 100),
                        Avg_Excess=("Top20 Excess %", "mean"),
                        Median_Excess=("Top20 Excess %", "median"),
                        Avg_Top20_Win=("Top20 Win %", "mean"),
                        Avg_Top20_Hit5=("Top20 Hit 5% %", "mean"),
                    )
                    .reset_index()
                )

                for frame in (vol_summary, beta_summary):
                    for col in ["Positive_Breadth", "Avg_Excess", "Median_Excess",
                                "Avg_Top20_Win", "Avg_Top20_Hit5"]:
                        frame[col] = frame[col].round(2)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**B. Performance by volatility group**")
                    st.dataframe(vol_summary, use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("**C. Performance by beta group**")
                    st.dataframe(beta_summary, use_container_width=True, hide_index=True)

            # Momentum/trend context using current characteristics as a diagnostic clue.
            momentum_source = diag.dropna(subset=["60D Return %"]).copy()
            if len(momentum_source) >= 12:
                try:
                    momentum_source["60D Momentum Group"] = pd.qcut(
                        momentum_source["60D Return %"],
                        q=3,
                        labels=["Low momentum", "Mid momentum", "High momentum"],
                        duplicates="drop",
                    )
                except Exception:
                    momentum_source["60D Momentum Group"] = "All"

                mom_summary = (
                    momentum_source.groupby("60D Momentum Group", observed=True)
                    .agg(
                        Tickers=("Ticker", "count"),
                        Positive_Breadth=("Top20 Excess %", lambda x: (x > 0).mean() * 100),
                        Avg_Excess=("Top20 Excess %", "mean"),
                        Median_Excess=("Top20 Excess %", "median"),
                        Avg_Top20_Win=("Top20 Win %", "mean"),
                        Avg_Top20_Hit5=("Top20 Hit 5% %", "mean"),
                    )
                    .reset_index()
                )
                for col in ["Positive_Breadth", "Avg_Excess", "Median_Excess",
                            "Avg_Top20_Win", "Avg_Top20_Hit5"]:
                    mom_summary[col] = mom_summary[col].round(2)

                st.markdown("**D. Performance by 60-day momentum group**")
                st.dataframe(mom_summary, use_container_width=True, hide_index=True)

            st.markdown("**E. Ticker diagnostic table**")
            st.dataframe(
                diag.sort_values("Top20 Excess %", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Phase 3F diagnostics CSV",
                diag.to_csv(index=False),
                "bullseye_phase3f_diagnostics.csv",
                "text/csv",
            )
        else:
            st.warning("No Phase 3F diagnostic results were returned.")


if run_point_in_time:
    with st.spinner("Running Phase 3G point-in-time validation..."):
        tickers2 = sorted(set(tickers + ["SPY"]))
        data = download_prices(tickers2)
        spy = one_symbol(data, "SPY")
        pit_rows = []

        if spy is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                pit_rows.extend(
                    point_in_time_backtest_symbol(
                        df,
                        spy,
                        t,
                        lookback_days=1260,
                        step=20,
                    )
                )

        if pit_rows:
            pit = pd.DataFrame(pit_rows)

            st.subheader("🕰️ Phase 3G Point-in-Time Validation")
            st.caption(
                "All stock characteristics below are calculated at each historical snapshot. "
                "No current-day beta, volatility, liquidity, or momentum data are used."
            )

            # Focus on the top 20% Experimental 3.0 observations versus all observations.
            q80 = pit["Experimental 3.0 Score"].quantile(0.80)
            top20 = pit[pit["Experimental 3.0 Score"] >= q80].copy()

            overall_compare = pd.DataFrame([
                {
                    "Group": "All snapshots",
                    "Samples": len(pit),
                    "Avg 20D %": round(pit["20D Forward %"].mean(), 2),
                    "20D Win %": round((pit["20D Forward %"] > 0).mean() * 100, 2),
                    "20D Hit 5% %": round((pit["20D Forward %"] >= 5).mean() * 100, 2),
                },
                {
                    "Group": "Top 20% Experimental 3.0",
                    "Samples": len(top20),
                    "Avg 20D %": round(top20["20D Forward %"].mean(), 2),
                    "20D Win %": round((top20["20D Forward %"] > 0).mean() * 100, 2),
                    "20D Hit 5% %": round((top20["20D Forward %"] >= 5).mean() * 100, 2),
                },
            ])
            st.markdown("**A. Point-in-time baseline vs top 20% Experimental 3.0**")
            st.dataframe(overall_compare, use_container_width=True, hide_index=True)

            # Study 120D trend strength.
            trend120 = pit.copy()
            trend120["120D Trend Group"] = pd.cut(
                trend120["120D Return %"],
                bins=[-1000, 0, 15, 30, 50, 1000],
                labels=["<0%", "0–15%", "15–30%", "30–50%", "50%+"],
                include_lowest=True,
            )
            trend120_summary = (
                trend120.groupby("120D Trend Group", observed=True)
                .agg(
                    Samples=("Ticker", "count"),
                    Avg_20D=("20D Forward %", "mean"),
                    Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                    Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                    Avg_Exp3=("Experimental 3.0 Score", "mean"),
                )
                .reset_index()
            )

            # Study 60D cooling / consolidation.
            trend60 = pit.copy()
            trend60["60D Momentum Group"] = pd.cut(
                trend60["60D Return %"],
                bins=[-1000, -5, 5, 15, 30, 1000],
                labels=["<-5%", "-5–5%", "5–15%", "15–30%", "30%+"],
                include_lowest=True,
            )
            trend60_summary = (
                trend60.groupby("60D Momentum Group", observed=True)
                .agg(
                    Samples=("Ticker", "count"),
                    Avg_20D=("20D Forward %", "mean"),
                    Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                    Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                    Avg_Exp3=("Experimental 3.0 Score", "mean"),
                )
                .reset_index()
            )

            for frame in (trend120_summary, trend60_summary):
                for col in ["Avg_20D", "Win_20D", "Hit_5pct_20D", "Avg_Exp3"]:
                    frame[col] = frame[col].round(2)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**B. 120-day trend performance**")
                st.dataframe(trend120_summary, use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**C. 60-day momentum / cooling performance**")
                st.dataframe(trend60_summary, use_container_width=True, hide_index=True)

            # Test the specific hypothesis:
            # strong 120D trend + moderate 60D momentum + positive 20MA position.
            hypothesis = pit.copy()
            hypothesis["Continuation Setup"] = (
                (hypothesis["120D Return %"] >= 30)
                & (hypothesis["60D Return %"] >= -5)
                & (hypothesis["60D Return %"] <= 15)
                & (hypothesis["Dist 20MA %"] > 0)
            )

            hypothesis_rows = []
            for name, grp in [
                ("Continuation setup = YES", hypothesis[hypothesis["Continuation Setup"]]),
                ("Continuation setup = NO", hypothesis[~hypothesis["Continuation Setup"]]),
            ]:
                if len(grp) == 0:
                    continue
                hypothesis_rows.append({
                    "Group": name,
                    "Samples": len(grp),
                    "Avg Experimental 3.0": round(grp["Experimental 3.0 Score"].mean(), 2),
                    "Avg 5D %": round(grp["5D Forward %"].mean(), 2),
                    "Avg 10D %": round(grp["10D Forward %"].mean(), 2),
                    "Avg 20D %": round(grp["20D Forward %"].mean(), 2),
                    "20D Win %": round((grp["20D Forward %"] > 0).mean() * 100, 2),
                    "20D Hit 5% %": round((grp["20D Forward %"] >= 5).mean() * 100, 2),
                })

            st.markdown("**D. Continuation-setup hypothesis test**")
            st.dataframe(pd.DataFrame(hypothesis_rows), use_container_width=True, hide_index=True)

            # Beta, volatility, and liquidity point-in-time buckets.
            bucket_rows = []
            bucket_specs = [
                ("Beta vs SPY", "Beta"),
                ("Ann Vol %", "Volatility"),
                ("Avg $ Volume 60D ($M)", "Liquidity"),
            ]

            for col, label in bucket_specs:
                temp = pit[[col, "20D Forward %", "Experimental 3.0 Score"]].dropna().copy()
                if len(temp) < 50 or temp[col].nunique() < 3:
                    continue
                try:
                    temp["Bucket"] = pd.qcut(
                        temp[col],
                        q=3,
                        labels=["Low", "Mid", "High"],
                        duplicates="drop",
                    )
                except Exception:
                    continue

                grouped = (
                    temp.groupby("Bucket", observed=True)
                    .agg(
                        Samples=(col, "count"),
                        Avg_Value=(col, "mean"),
                        Avg_Exp3=("Experimental 3.0 Score", "mean"),
                        Avg_20D=("20D Forward %", "mean"),
                        Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                        Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                    )
                    .reset_index()
                )
                for _, r in grouped.iterrows():
                    bucket_rows.append({
                        "Characteristic": label,
                        "Bucket": r["Bucket"],
                        "Samples": int(r["Samples"]),
                        "Avg Value": round(r["Avg_Value"], 2),
                        "Avg Experimental 3.0": round(r["Avg_Exp3"], 2),
                        "Avg 20D %": round(r["Avg_20D"], 2),
                        "20D Win %": round(r["Win_20D"], 2),
                        "20D Hit 5% %": round(r["Hit_5pct_20D"], 2),
                    })

            if bucket_rows:
                st.markdown("**E. Point-in-time beta / volatility / liquidity study**")
                st.dataframe(
                    pd.DataFrame(bucket_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("**F. Point-in-time historical snapshots**")
            st.dataframe(
                pit.sort_values(["Date", "Experimental 3.0 Score"], ascending=[False, False]),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Phase 3G point-in-time CSV",
                pit.to_csv(index=False),
                "bullseye_phase3g_point_in_time.csv",
                "text/csv",
            )
        else:
            st.warning("No Phase 3G point-in-time samples were returned.")


if run_interactions:
    with st.spinner("Running Phase 3H interaction testing..."):
        tickers2 = sorted(set(tickers + ["SPY"]))
        data = download_prices(tickers2)
        spy = one_symbol(data, "SPY")
        interaction_rows = []

        if spy is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                interaction_rows.extend(
                    point_in_time_backtest_symbol(
                        df,
                        spy,
                        t,
                        lookback_days=1260,
                        step=20,
                    )
                )

        if interaction_rows:
            it = pd.DataFrame(interaction_rows).dropna(
                subset=["Experimental 3.0 Score", "Beta vs SPY", "Ann Vol %", "20D Forward %"]
            ).copy()

            st.subheader("🧩 Phase 3H Interaction Test")
            st.caption(
                "Experimental 3.0 remains frozen. This test asks whether top-ranked 3.0 setups "
                "perform better when point-in-time beta and/or volatility are also elevated."
            )

            # Global thresholds determined from the historical sample.
            exp80 = it["Experimental 3.0 Score"].quantile(0.80)
            exp90 = it["Experimental 3.0 Score"].quantile(0.90)
            beta67 = it["Beta vs SPY"].quantile(0.67)
            vol67 = it["Ann Vol %"].quantile(0.67)

            it["Top20 Exp3"] = it["Experimental 3.0 Score"] >= exp80
            it["Top10 Exp3"] = it["Experimental 3.0 Score"] >= exp90
            it["High Beta"] = it["Beta vs SPY"] >= beta67
            it["High Vol"] = it["Ann Vol %"] >= vol67

            # Core interaction groups.
            groups = [
                ("All snapshots", it),
                ("Top 20% Exp3", it[it["Top20 Exp3"]]),
                ("Top 10% Exp3", it[it["Top10 Exp3"]]),
                ("High beta only", it[it["High Beta"]]),
                ("High volatility only", it[it["High Vol"]]),
                ("Top20 Exp3 + high beta", it[it["Top20 Exp3"] & it["High Beta"]]),
                ("Top20 Exp3 + high vol", it[it["Top20 Exp3"] & it["High Vol"]]),
                ("Top20 Exp3 + high beta + high vol",
                 it[it["Top20 Exp3"] & it["High Beta"] & it["High Vol"]]),
                ("Top10 Exp3 + high beta", it[it["Top10 Exp3"] & it["High Beta"]]),
                ("Top10 Exp3 + high vol", it[it["Top10 Exp3"] & it["High Vol"]]),
                ("Top10 Exp3 + high beta + high vol",
                 it[it["Top10 Exp3"] & it["High Beta"] & it["High Vol"]]),
            ]

            rows = []
            baseline_20d = it["20D Forward %"].mean()

            for name, grp in groups:
                if len(grp) < 20:
                    continue
                rows.append({
                    "Group": name,
                    "Samples": len(grp),
                    "Avg Exp3": round(grp["Experimental 3.0 Score"].mean(), 2),
                    "Avg Beta": round(grp["Beta vs SPY"].mean(), 2),
                    "Avg Vol %": round(grp["Ann Vol %"].mean(), 2),
                    "Avg 5D %": round(grp["5D Forward %"].mean(), 2),
                    "Avg 10D %": round(grp["10D Forward %"].mean(), 2),
                    "Avg 20D %": round(grp["20D Forward %"].mean(), 2),
                    "Excess vs All 20D %": round(grp["20D Forward %"].mean() - baseline_20d, 2),
                    "20D Win %": round((grp["20D Forward %"] > 0).mean() * 100, 2),
                    "20D Hit 5% %": round((grp["20D Forward %"] >= 5).mean() * 100, 2),
                    "20D Hit 10% %": round((grp["20D Forward %"] >= 10).mean() * 100, 2),
                })

            interaction_summary = pd.DataFrame(rows).sort_values(
                "Avg 20D %", ascending=False
            )

            st.markdown("**A. Core interaction results**")
            st.dataframe(
                interaction_summary,
                use_container_width=True,
                hide_index=True,
            )

            # Compare top 20% Experimental 3.0 across beta x volatility cells.
            beta_labels = ["Low beta", "Mid beta", "High beta"]
            vol_labels = ["Low vol", "Mid vol", "High vol"]

            try:
                it["Beta Tercile"] = pd.qcut(
                    it["Beta vs SPY"],
                    q=3,
                    labels=beta_labels,
                    duplicates="drop",
                )
                it["Vol Tercile"] = pd.qcut(
                    it["Ann Vol %"],
                    q=3,
                    labels=vol_labels,
                    duplicates="drop",
                )
            except Exception:
                it["Beta Tercile"] = "All"
                it["Vol Tercile"] = "All"

            top20 = it[it["Top20 Exp3"]].copy()
            matrix = (
                top20.groupby(["Beta Tercile", "Vol Tercile"], observed=True)
                .agg(
                    Samples=("Ticker", "count"),
                    Avg_20D=("20D Forward %", "mean"),
                    Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                    Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                    Hit_10pct_20D=("20D Forward %", lambda x: (x >= 10).mean() * 100),
                )
                .reset_index()
            )
            for col in ["Avg_20D", "Win_20D", "Hit_5pct_20D", "Hit_10pct_20D"]:
                matrix[col] = matrix[col].round(2)

            st.markdown("**B. Top-20% Experimental 3.0 by beta × volatility**")
            st.dataframe(matrix, use_container_width=True, hide_index=True)

            # Ticker breadth for the best-performing interaction, selected by average 20D
            # among groups with at least 40 observations.
            eligible = interaction_summary[interaction_summary["Samples"] >= 40].copy()
            if len(eligible):
                best_group_name = eligible.iloc[0]["Group"]

                mask_map = {
                    "All snapshots": pd.Series(True, index=it.index),
                    "Top 20% Exp3": it["Top20 Exp3"],
                    "Top 10% Exp3": it["Top10 Exp3"],
                    "High beta only": it["High Beta"],
                    "High volatility only": it["High Vol"],
                    "Top20 Exp3 + high beta": it["Top20 Exp3"] & it["High Beta"],
                    "Top20 Exp3 + high vol": it["Top20 Exp3"] & it["High Vol"],
                    "Top20 Exp3 + high beta + high vol":
                        it["Top20 Exp3"] & it["High Beta"] & it["High Vol"],
                    "Top10 Exp3 + high beta": it["Top10 Exp3"] & it["High Beta"],
                    "Top10 Exp3 + high vol": it["Top10 Exp3"] & it["High Vol"],
                    "Top10 Exp3 + high beta + high vol":
                        it["Top10 Exp3"] & it["High Beta"] & it["High Vol"],
                }

                best = it[mask_map[best_group_name]].copy()
                ticker_stats = (
                    best.groupby("Ticker", observed=True)
                    .agg(
                        Samples=("Ticker", "count"),
                        Avg_20D=("20D Forward %", "mean"),
                        Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                        Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                    )
                    .reset_index()
                )
                ticker_stats = ticker_stats[ticker_stats["Samples"] >= 2].copy()
                for col in ["Avg_20D", "Win_20D", "Hit_5pct_20D"]:
                    ticker_stats[col] = ticker_stats[col].round(2)

                if len(ticker_stats):
                    positive = int((ticker_stats["Avg_20D"] > 0).sum())
                    breadth = pd.DataFrame([{
                        "Best Interaction": best_group_name,
                        "Tickers Tested": len(ticker_stats),
                        "Tickers Avg 20D > 0": positive,
                        "Positive Ticker Breadth %": round(
                            positive / len(ticker_stats) * 100, 2
                        ),
                        "Median Ticker Avg 20D %": round(
                            ticker_stats["Avg_20D"].median(), 2
                        ),
                    }])

                    st.markdown("**C. Breadth of best interaction**")
                    st.dataframe(breadth, use_container_width=True, hide_index=True)

                    with st.expander("Ticker results for best interaction"):
                        st.dataframe(
                            ticker_stats.sort_values("Avg_20D", ascending=False),
                            use_container_width=True,
                            hide_index=True,
                        )

            st.markdown("**Thresholds used in this test**")
            thresholds = pd.DataFrame([{
                "Top20 Exp3 cutoff": round(exp80, 2),
                "Top10 Exp3 cutoff": round(exp90, 2),
                "High-beta cutoff": round(beta67, 2),
                "High-volatility cutoff": round(vol67, 2),
            }])
            st.dataframe(thresholds, use_container_width=True, hide_index=True)

            st.download_button(
                "Download Phase 3H interaction snapshots CSV",
                it.to_csv(index=False),
                "bullseye_phase3h_interactions.csv",
                "text/csv",
            )
        else:
            st.warning("No Phase 3H interaction samples were returned.")


if run_robustness:
    with st.spinner("Running Phase 3I robustness validation..."):
        tickers2 = sorted(set(tickers + ["SPY"]))
        data = download_prices(tickers2)
        spy = one_symbol(data, "SPY")
        robust_rows = []

        if spy is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                robust_rows.extend(
                    point_in_time_backtest_symbol(
                        df,
                        spy,
                        t,
                        lookback_days=1260,
                        step=20,
                    )
                )

        if robust_rows:
            rb = pd.DataFrame(robust_rows).dropna(
                subset=["Experimental 3.0 Score", "Beta vs SPY", "20D Forward %", "Date"]
            ).copy()
            rb["Date"] = pd.to_datetime(rb["Date"])

            st.subheader("🛡️ Phase 3I Robustness Validation")
            st.caption(
                "The Phase 3H winning interaction is frozen: Top-20% Experimental 3.0 + high beta. "
                "This test checks whether it survives separate time blocks and broad ticker-level testing."
            )

            exp80 = rb["Experimental 3.0 Score"].quantile(0.80)
            beta67 = rb["Beta vs SPY"].quantile(0.67)

            rb["Top20 Exp3"] = rb["Experimental 3.0 Score"] >= exp80
            rb["High Beta"] = rb["Beta vs SPY"] >= beta67
            rb["3I Signal"] = rb["Top20 Exp3"] & rb["High Beta"]

            # ------------------------------------------------------------
            # A) Separate chronological periods
            # ------------------------------------------------------------
            unique_dates = sorted(rb["Date"].unique())
            cuts = np.array_split(np.array(unique_dates), 3)
            period_names = ["Older period", "Middle period", "Recent period"]

            period_rows = []
            for period_name, dates in zip(period_names, cuts):
                if len(dates) == 0:
                    continue
                start_date = pd.Timestamp(dates[0])
                end_date = pd.Timestamp(dates[-1])
                block = rb[(rb["Date"] >= start_date) & (rb["Date"] <= end_date)].copy()

                groups = [
                    ("All snapshots", block),
                    ("Top20 Exp3", block[block["Top20 Exp3"]]),
                    ("High beta", block[block["High Beta"]]),
                    ("Top20 Exp3 + high beta", block[block["3I Signal"]]),
                ]

                for group_name, grp in groups:
                    if len(grp) < 10:
                        continue
                    period_rows.append({
                        "Period": period_name,
                        "Start": start_date.date(),
                        "End": end_date.date(),
                        "Group": group_name,
                        "Samples": len(grp),
                        "Avg 5D %": round(grp["5D Forward %"].mean(), 2),
                        "Avg 10D %": round(grp["10D Forward %"].mean(), 2),
                        "Avg 20D %": round(grp["20D Forward %"].mean(), 2),
                        "20D Win %": round((grp["20D Forward %"] > 0).mean() * 100, 2),
                        "20D Hit 5% %": round((grp["20D Forward %"] >= 5).mean() * 100, 2),
                        "20D Hit 10% %": round((grp["20D Forward %"] >= 10).mean() * 100, 2),
                    })

            period_df = pd.DataFrame(period_rows)
            st.markdown("**A. Winning interaction by separate historical period**")
            st.dataframe(period_df, use_container_width=True, hide_index=True)

            # Summary for the frozen interaction only.
            sig_periods = period_df[period_df["Group"] == "Top20 Exp3 + high beta"].copy()
            if len(sig_periods):
                sig_summary = pd.DataFrame([{
                    "Periods Tested": sig_periods["Period"].nunique(),
                    "Avg Period 20D %": round(sig_periods["Avg 20D %"].mean(), 2),
                    "Worst Period 20D %": round(sig_periods["Avg 20D %"].min(), 2),
                    "Best Period 20D %": round(sig_periods["Avg 20D %"].max(), 2),
                    "Positive Periods": int((sig_periods["Avg 20D %"] > 0).sum()),
                    "Avg Period Win %": round(sig_periods["20D Win %"].mean(), 2),
                    "Avg Period Hit 5%": round(sig_periods["20D Hit 5% %"].mean(), 2),
                }])
                st.markdown("**B. Frozen interaction period summary**")
                st.dataframe(sig_summary, use_container_width=True, hide_index=True)

            # ------------------------------------------------------------
            # C) Ticker-level robustness
            # ------------------------------------------------------------
            ticker_rows = []
            for ticker, grp in rb.groupby("Ticker", observed=True):
                all_grp = grp.copy()
                sig = grp[grp["3I Signal"]].copy()

                if len(all_grp) < 8 or len(sig) < 2:
                    continue

                ticker_rows.append({
                    "Ticker": ticker,
                    "All Samples": len(all_grp),
                    "Signal Samples": len(sig),
                    "All Avg 20D %": round(all_grp["20D Forward %"].mean(), 2),
                    "Signal Avg 20D %": round(sig["20D Forward %"].mean(), 2),
                    "Signal Excess %": round(
                        sig["20D Forward %"].mean() - all_grp["20D Forward %"].mean(), 2
                    ),
                    "Signal Win %": round((sig["20D Forward %"] > 0).mean() * 100, 2),
                    "Signal Hit 5% %": round((sig["20D Forward %"] >= 5).mean() * 100, 2),
                    "Signal Hit 10% %": round((sig["20D Forward %"] >= 10).mean() * 100, 2),
                })

            ticker_df = pd.DataFrame(ticker_rows)

            if len(ticker_df):
                positive_excess = int((ticker_df["Signal Excess %"] > 0).sum())
                positive_returns = int((ticker_df["Signal Avg 20D %"] > 0).sum())
                tested = len(ticker_df)

                breadth_summary = pd.DataFrame([{
                    "Tickers Tested": tested,
                    "Positive Excess Tickers": positive_excess,
                    "Positive Excess Breadth %": round(
                        positive_excess / tested * 100, 2
                    ) if tested else np.nan,
                    "Positive Return Tickers": positive_returns,
                    "Positive Return Breadth %": round(
                        positive_returns / tested * 100, 2
                    ) if tested else np.nan,
                    "Median Signal Avg 20D %": round(ticker_df["Signal Avg 20D %"].median(), 2),
                    "Median Signal Excess %": round(ticker_df["Signal Excess %"].median(), 2),
                }])

                st.markdown("**C. Ticker-level robustness summary**")
                st.dataframe(breadth_summary, use_container_width=True, hide_index=True)

                st.markdown("**D. Ticker-by-ticker robustness**")
                st.dataframe(
                    ticker_df.sort_values("Signal Excess %", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

            # ------------------------------------------------------------
            # E) Stronger vs weaker market environment
            # ------------------------------------------------------------
            rb["Regime Group"] = np.where(
                rb["Market Regime"] >= 7,
                "Stronger market",
                "Weaker market",
            )

            regime_rows = []
            for regime_name, grp in rb.groupby("Regime Group", observed=True):
                for group_name, subset in [
                    ("All", grp),
                    ("Top20 Exp3 + high beta", grp[grp["3I Signal"]]),
                ]:
                    if len(subset) < 10:
                        continue
                    regime_rows.append({
                        "Market Regime": regime_name,
                        "Group": group_name,
                        "Samples": len(subset),
                        "Avg 5D %": round(subset["5D Forward %"].mean(), 2),
                        "Avg 10D %": round(subset["10D Forward %"].mean(), 2),
                        "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                        "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                        "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                        "20D Hit 10% %": round((subset["20D Forward %"] >= 10).mean() * 100, 2),
                    })

            st.markdown("**E. Frozen interaction by market regime**")
            st.dataframe(
                pd.DataFrame(regime_rows),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**Thresholds frozen from Phase 3H logic**")
            st.dataframe(
                pd.DataFrame([{
                    "Top20 Experimental 3.0 cutoff": round(exp80, 2),
                    "High-beta cutoff": round(beta67, 2),
                }]),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Phase 3I robustness CSV",
                rb.to_csv(index=False),
                "bullseye_phase3i_robustness.csv",
                "text/csv",
            )
        else:
            st.warning("No Phase 3I robustness samples were returned.")


if run_phase4a:
    with st.spinner("Running Bullseye 4.0 prototype validation..."):
        tickers2 = sorted(set(tickers + ["SPY"]))
        data = download_prices(tickers2)
        spy = one_symbol(data, "SPY")
        p4_rows = []

        if spy is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                p4_rows.extend(
                    point_in_time_backtest_symbol(
                        df,
                        spy,
                        t,
                        lookback_days=1260,
                        step=20,
                    )
                )

        if p4_rows:
            p4 = pd.DataFrame(p4_rows).dropna(
                subset=["Bullseye 4.0 Score", "Experimental 3.0 Score", "20D Forward %", "Date"]
            ).copy()
            p4["Date"] = pd.to_datetime(p4["Date"])

            st.subheader("🚀 Phase 4A Bullseye 4.0 Prototype")
            st.caption(
                "Bullseye 4.0 is being tested beside Experimental 3.0. "
                "The live scanner is not being permanently switched yet."
            )

            # A) Four-system head-to-head.
            system_rows = []
            score_systems = [
                "Bullseye Score",
                "Opportunity Score",
                "Experimental 3.0 Score",
                "Bullseye 4.0 Score",
            ]

            for score_name in score_systems:
                temp = p4[
                    [score_name, "5D Forward %", "10D Forward %", "15D Forward %", "20D Forward %"]
                ].dropna()
                if len(temp) < 50:
                    continue

                q80 = temp[score_name].quantile(0.80)
                q90 = temp[score_name].quantile(0.90)

                for group_name, subset in [
                    ("All", temp),
                    ("Top 20%", temp[temp[score_name] >= q80]),
                    ("Top 10%", temp[temp[score_name] >= q90]),
                ]:
                    if len(subset) == 0:
                        continue
                    system_rows.append({
                        "Score System": score_name,
                        "Group": group_name,
                        "Samples": len(subset),
                        "Avg 5D %": round(subset["5D Forward %"].mean(), 2),
                        "Avg 10D %": round(subset["10D Forward %"].mean(), 2),
                        "Avg 15D %": round(subset["15D Forward %"].mean(), 2),
                        "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                        "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                        "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                        "20D Hit 10% %": round((subset["20D Forward %"] >= 10).mean() * 100, 2),
                    })

            system_df = pd.DataFrame(system_rows)
            st.markdown("**A. Bullseye 4.0 head-to-head**")
            st.dataframe(system_df, use_container_width=True, hide_index=True)

            # B) 4.0 score buckets.
            p4["4.0 Bucket"] = pd.cut(
                p4["Bullseye 4.0 Score"],
                bins=[-0.01, 59.99, 69.99, 79.99, 89.99, 100],
                labels=["<60", "60–69.9", "70–79.9", "80–89.9", "90+"],
            )
            bucket_df = (
                p4.groupby("4.0 Bucket", observed=True)
                .agg(
                    Samples=("Ticker", "count"),
                    Avg_5D=("5D Forward %", "mean"),
                    Avg_10D=("10D Forward %", "mean"),
                    Avg_20D=("20D Forward %", "mean"),
                    Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                    Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                    Hit_10pct_20D=("20D Forward %", lambda x: (x >= 10).mean() * 100),
                )
                .reset_index()
            )
            for col in ["Avg_5D", "Avg_10D", "Avg_20D", "Win_20D",
                        "Hit_5pct_20D", "Hit_10pct_20D"]:
                bucket_df[col] = bucket_df[col].round(2)

            st.markdown("**B. Bullseye 4.0 score buckets**")
            st.dataframe(bucket_df, use_container_width=True, hide_index=True)

            # C) Separate chronological periods for top-20% 4.0 vs 3.0.
            unique_dates = sorted(p4["Date"].unique())
            cuts = np.array_split(np.array(unique_dates), 3)
            period_names = ["Older period", "Middle period", "Recent period"]
            period_rows = []

            for period_name, dates in zip(period_names, cuts):
                if len(dates) == 0:
                    continue
                start_date = pd.Timestamp(dates[0])
                end_date = pd.Timestamp(dates[-1])
                block = p4[(p4["Date"] >= start_date) & (p4["Date"] <= end_date)].copy()

                for score_name in ["Experimental 3.0 Score", "Bullseye 4.0 Score"]:
                    temp = block[[score_name, "20D Forward %"]].dropna()
                    if len(temp) < 20:
                        continue
                    q80 = temp[score_name].quantile(0.80)
                    top = temp[temp[score_name] >= q80]
                    period_rows.append({
                        "Period": period_name,
                        "Start": start_date.date(),
                        "End": end_date.date(),
                        "Score System": score_name,
                        "Top20 Samples": len(top),
                        "Top20 Avg 20D %": round(top["20D Forward %"].mean(), 2),
                        "Top20 Win %": round((top["20D Forward %"] > 0).mean() * 100, 2),
                        "Top20 Hit 5% %": round((top["20D Forward %"] >= 5).mean() * 100, 2),
                    })

            period_df = pd.DataFrame(period_rows)
            st.markdown("**C. Experimental 3.0 vs Bullseye 4.0 by historical period**")
            st.dataframe(period_df, use_container_width=True, hide_index=True)

            # D) Accelerator diagnostics.
            accel_df = p4.copy()
            accel_df["Accelerator Group"] = pd.cut(
                accel_df["4.0 Accelerator"],
                bins=[-0.01, 0.01, 3.99, 7.99, 12.01],
                labels=["None", "Light", "Moderate", "Strong"],
                include_lowest=True,
            )
            accel_summary = (
                accel_df.groupby("Accelerator Group", observed=True)
                .agg(
                    Samples=("Ticker", "count"),
                    Avg_Exp3=("Experimental 3.0 Score", "mean"),
                    Avg_4_0=("Bullseye 4.0 Score", "mean"),
                    Avg_20D=("20D Forward %", "mean"),
                    Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                    Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                )
                .reset_index()
            )
            for col in ["Avg_Exp3", "Avg_4_0", "Avg_20D", "Win_20D", "Hit_5pct_20D"]:
                accel_summary[col] = accel_summary[col].round(2)

            st.markdown("**D. 4.0 accelerator diagnostics**")
            st.dataframe(accel_summary, use_container_width=True, hide_index=True)

            st.download_button(
                "Download Phase 4A prototype CSV",
                p4.to_csv(index=False),
                "bullseye_phase4a_prototype.csv",
                "text/csv",
            )
        else:
            st.warning("No Phase 4A prototype samples were returned.")


if run_phase4b:
    with st.spinner("Running Phase 4B accelerator tuning..."):
        tickers2 = sorted(set(tickers + ["SPY"]))
        data = download_prices(tickers2)
        spy = one_symbol(data, "SPY")
        tune_rows = []

        if spy is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                tune_rows.extend(
                    point_in_time_backtest_symbol(
                        df,
                        spy,
                        t,
                        lookback_days=1260,
                        step=20,
                    )
                )

        if tune_rows:
            tune = pd.DataFrame(tune_rows).dropna(
                subset=[
                    "Experimental 3.0 Score",
                    "Bullseye 4.0 Score",
                    "Bullseye 4B1 Score",
                    "Bullseye 4B2 Score",
                    "Bullseye 4B3 Score",
                    "20D Forward %",
                    "Date",
                ]
            ).copy()
            tune["Date"] = pd.to_datetime(tune["Date"])

            st.subheader("🧪 Phase 4B Accelerator Tuning")
            st.caption(
                "4A remains unchanged. We are comparing stricter accelerator variants before choosing a winner."
            )

            systems = [
                "Experimental 3.0 Score",
                "Bullseye 4.0 Score",
                "Bullseye 4B1 Score",
                "Bullseye 4B2 Score",
                "Bullseye 4B3 Score",
            ]

            # A) Full-sample head-to-head
            rows = []
            for score_name in systems:
                temp = tune[
                    [score_name, "5D Forward %", "10D Forward %", "15D Forward %", "20D Forward %"]
                ].dropna()
                if len(temp) < 50:
                    continue

                q80 = temp[score_name].quantile(0.80)
                q90 = temp[score_name].quantile(0.90)

                for group_name, subset in [
                    ("Top 20%", temp[temp[score_name] >= q80]),
                    ("Top 10%", temp[temp[score_name] >= q90]),
                ]:
                    rows.append({
                        "Score System": score_name,
                        "Group": group_name,
                        "Samples": len(subset),
                        "Avg 5D %": round(subset["5D Forward %"].mean(), 2),
                        "Avg 10D %": round(subset["10D Forward %"].mean(), 2),
                        "Avg 15D %": round(subset["15D Forward %"].mean(), 2),
                        "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                        "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                        "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                        "20D Hit 10% %": round((subset["20D Forward %"] >= 10).mean() * 100, 2),
                    })

            head = pd.DataFrame(rows)
            st.markdown("**A. Accelerator-variant head-to-head**")
            st.dataframe(head, use_container_width=True, hide_index=True)

            # B) Separate periods, top 10% only
            unique_dates = sorted(tune["Date"].unique())
            cuts = np.array_split(np.array(unique_dates), 3)
            period_names = ["Older period", "Middle period", "Recent period"]
            period_rows = []

            for period_name, dates in zip(period_names, cuts):
                if len(dates) == 0:
                    continue
                start_date = pd.Timestamp(dates[0])
                end_date = pd.Timestamp(dates[-1])
                block = tune[(tune["Date"] >= start_date) & (tune["Date"] <= end_date)].copy()

                for score_name in systems:
                    temp = block[[score_name, "20D Forward %"]].dropna()
                    if len(temp) < 20:
                        continue
                    q90 = temp[score_name].quantile(0.90)
                    top = temp[temp[score_name] >= q90]

                    period_rows.append({
                        "Period": period_name,
                        "Score System": score_name,
                        "Samples": len(top),
                        "Top10 Avg 20D %": round(top["20D Forward %"].mean(), 2),
                        "Top10 Win %": round((top["20D Forward %"] > 0).mean() * 100, 2),
                        "Top10 Hit 5% %": round((top["20D Forward %"] >= 5).mean() * 100, 2),
                        "Top10 Hit 10% %": round((top["20D Forward %"] >= 10).mean() * 100, 2),
                    })

            period_df = pd.DataFrame(period_rows)
            st.markdown("**B. Top-10% performance by historical period**")
            st.dataframe(period_df, use_container_width=True, hide_index=True)

            # C) Compact robustness scorecard
            scorecard_rows = []
            for score_name in systems:
                temp = period_df[period_df["Score System"] == score_name]
                if len(temp) == 0:
                    continue
                scorecard_rows.append({
                    "Score System": score_name,
                    "Periods": len(temp),
                    "Avg Period 20D %": round(temp["Top10 Avg 20D %"].mean(), 2),
                    "Worst Period 20D %": round(temp["Top10 Avg 20D %"].min(), 2),
                    "Best Period 20D %": round(temp["Top10 Avg 20D %"].max(), 2),
                    "Positive Periods": int((temp["Top10 Avg 20D %"] > 0).sum()),
                    "Avg Win %": round(temp["Top10 Win %"].mean(), 2),
                    "Avg Hit 5%": round(temp["Top10 Hit 5% %"].mean(), 2),
                })

            scorecard = pd.DataFrame(scorecard_rows).sort_values(
                ["Avg Period 20D %", "Worst Period 20D %"],
                ascending=[False, False],
            )

            st.markdown("**C. Robustness scorecard**")
            st.dataframe(scorecard, use_container_width=True, hide_index=True)

            # D) Accelerator usage
            usage = pd.DataFrame([
                {
                    "Variant": "4A current",
                    "Boosted Samples": int((tune["4.0 Accelerator"] > 0).sum()),
                    "Moderate+ Samples": int((tune["4.0 Accelerator"] >= 4).sum()),
                    "Strong Samples": int((tune["4.0 Accelerator"] >= 8).sum()),
                },
                {
                    "Variant": "4B1 no-light",
                    "Boosted Samples": int((tune["4B1 Accelerator"] > 0).sum()),
                    "Moderate+ Samples": int((tune["4B1 Accelerator"] >= 4).sum()),
                    "Strong Samples": int((tune["4B1 Accelerator"] >= 8).sum()),
                },
                {
                    "Variant": "4B2 strict",
                    "Boosted Samples": int((tune["4B2 Accelerator"] > 0).sum()),
                    "Moderate+ Samples": int((tune["4B2 Accelerator"] >= 4).sum()),
                    "Strong Samples": int((tune["4B2 Accelerator"] >= 8).sum()),
                },
                {
                    "Variant": "4B3 moderate-only",
                    "Boosted Samples": int((tune["4B3 Accelerator"] > 0).sum()),
                    "Moderate+ Samples": int((tune["4B3 Accelerator"] >= 4).sum()),
                    "Strong Samples": int((tune["4B3 Accelerator"] >= 8).sum()),
                },
            ])

            st.markdown("**D. Accelerator usage by variant**")
            st.dataframe(usage, use_container_width=True, hide_index=True)

            st.download_button(
                "Download Phase 4B tuning CSV",
                tune.to_csv(index=False),
                "bullseye_phase4b_tuning.csv",
                "text/csv",
            )
        else:
            st.warning("No Phase 4B tuning samples were returned.")


if run_phase4c:
    with st.spinner("Running Phase 4C broad-universe validation..."):
        broad_tickers = sorted(set(BROAD_TICKERS))
        tickers2 = sorted(set(broad_tickers + ["SPY"]))
        data = download_prices(tickers2)
        spy = one_symbol(data, "SPY")
        broad_rows = []

        if spy is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in broad_tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                broad_rows.extend(
                    point_in_time_backtest_symbol(
                        df,
                        spy,
                        t,
                        lookback_days=1260,
                        step=20,
                    )
                )

        if broad_rows:
            broad = pd.DataFrame(broad_rows).dropna(
                subset=["Bullseye 4.0 Score", "Experimental 3.0 Score", "20D Forward %", "Date"]
            ).copy()
            broad["Date"] = pd.to_datetime(broad["Date"])

            st.subheader("🌐 Phase 4C Broad-Universe Validation")
            st.caption(
                "Bullseye 4.0 is frozen exactly as selected after Phase 4B. "
                "This test expands the universe across technology, financials, energy, healthcare, "
                "industrials, consumer, communications, utilities, materials, and real estate."
            )

            # A) Universe coverage
            coverage = pd.DataFrame([{
                "Requested Tickers": len(broad_tickers),
                "Tickers With Usable History": broad["Ticker"].nunique(),
                "Historical Snapshots": len(broad),
                "First Snapshot": broad["Date"].min().date(),
                "Last Snapshot": broad["Date"].max().date(),
            }])
            st.markdown("**A. Broad-universe coverage**")
            st.dataframe(coverage, use_container_width=True, hide_index=True)

            # B) Head-to-head across the broad universe
            score_systems = [
                "Bullseye Score",
                "Opportunity Score",
                "Experimental 3.0 Score",
                "Bullseye 4.0 Score",
            ]

            head_rows = []
            for score_name in score_systems:
                temp = broad[
                    [score_name, "5D Forward %", "10D Forward %", "15D Forward %", "20D Forward %"]
                ].dropna()
                if len(temp) < 100:
                    continue

                q80 = temp[score_name].quantile(0.80)
                q90 = temp[score_name].quantile(0.90)

                for group_name, subset in [
                    ("All", temp),
                    ("Top 20%", temp[temp[score_name] >= q80]),
                    ("Top 10%", temp[temp[score_name] >= q90]),
                ]:
                    head_rows.append({
                        "Score System": score_name,
                        "Group": group_name,
                        "Samples": len(subset),
                        "Avg 5D %": round(subset["5D Forward %"].mean(), 2),
                        "Avg 10D %": round(subset["10D Forward %"].mean(), 2),
                        "Avg 15D %": round(subset["15D Forward %"].mean(), 2),
                        "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                        "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                        "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                        "20D Hit 10% %": round((subset["20D Forward %"] >= 10).mean() * 100, 2),
                    })

            head_df = pd.DataFrame(head_rows)
            st.markdown("**B. Broad-universe score-system head-to-head**")
            st.dataframe(head_df, use_container_width=True, hide_index=True)

            # C) Bullseye 4.0 score buckets
            broad["4.0 Bucket"] = pd.cut(
                broad["Bullseye 4.0 Score"],
                bins=[-0.01, 59.99, 69.99, 79.99, 89.99, 100],
                labels=["<60", "60–69.9", "70–79.9", "80–89.9", "90+"],
            )

            bucket_df = (
                broad.groupby("4.0 Bucket", observed=True)
                .agg(
                    Samples=("Ticker", "count"),
                    Tickers=("Ticker", "nunique"),
                    Avg_5D=("5D Forward %", "mean"),
                    Avg_10D=("10D Forward %", "mean"),
                    Avg_20D=("20D Forward %", "mean"),
                    Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                    Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                    Hit_10pct_20D=("20D Forward %", lambda x: (x >= 10).mean() * 100),
                )
                .reset_index()
            )
            for col in ["Avg_5D", "Avg_10D", "Avg_20D", "Win_20D", "Hit_5pct_20D", "Hit_10pct_20D"]:
                bucket_df[col] = bucket_df[col].round(2)

            st.markdown("**C. Bullseye 4.0 score buckets — broad universe**")
            st.dataframe(bucket_df, use_container_width=True, hide_index=True)

            # D) Separate time periods for 4.0 vs 3.0
            unique_dates = sorted(broad["Date"].unique())
            cuts = np.array_split(np.array(unique_dates), 3)
            period_names = ["Older period", "Middle period", "Recent period"]
            period_rows = []

            for period_name, dates in zip(period_names, cuts):
                if len(dates) == 0:
                    continue
                start_date = pd.Timestamp(dates[0])
                end_date = pd.Timestamp(dates[-1])
                block = broad[(broad["Date"] >= start_date) & (broad["Date"] <= end_date)].copy()

                for score_name in ["Experimental 3.0 Score", "Bullseye 4.0 Score"]:
                    temp = block[[score_name, "20D Forward %"]].dropna()
                    if len(temp) < 50:
                        continue
                    q80 = temp[score_name].quantile(0.80)
                    q90 = temp[score_name].quantile(0.90)

                    for group_name, subset in [
                        ("Top 20%", temp[temp[score_name] >= q80]),
                        ("Top 10%", temp[temp[score_name] >= q90]),
                    ]:
                        period_rows.append({
                            "Period": period_name,
                            "Score System": score_name,
                            "Group": group_name,
                            "Samples": len(subset),
                            "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                            "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                            "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                            "20D Hit 10% %": round((subset["20D Forward %"] >= 10).mean() * 100, 2),
                        })

            period_df = pd.DataFrame(period_rows)
            st.markdown("**D. 3.0 vs 4.0 by broad-universe historical period**")
            st.dataframe(period_df, use_container_width=True, hide_index=True)

            # E) Ticker breadth for the top 20% Bullseye 4.0 signal
            q80_4 = broad["Bullseye 4.0 Score"].quantile(0.80)
            ticker_rows = []

            for ticker, grp in broad.groupby("Ticker", observed=True):
                grp = grp.dropna(subset=["Bullseye 4.0 Score", "20D Forward %"]).copy()
                if len(grp) < 8:
                    continue

                top = grp[grp["Bullseye 4.0 Score"] >= q80_4]
                if len(top) < 2:
                    continue

                ticker_rows.append({
                    "Ticker": ticker,
                    "All Samples": len(grp),
                    "Top20 Samples": len(top),
                    "All Avg 20D %": round(grp["20D Forward %"].mean(), 2),
                    "Top20 Avg 20D %": round(top["20D Forward %"].mean(), 2),
                    "Top20 Excess %": round(
                        top["20D Forward %"].mean() - grp["20D Forward %"].mean(), 2
                    ),
                    "Top20 Win %": round((top["20D Forward %"] > 0).mean() * 100, 2),
                    "Top20 Hit 5% %": round((top["20D Forward %"] >= 5).mean() * 100, 2),
                })

            ticker_df = pd.DataFrame(ticker_rows)
            if len(ticker_df):
                positive_excess = int((ticker_df["Top20 Excess %"] > 0).sum())
                positive_return = int((ticker_df["Top20 Avg 20D %"] > 0).sum())
                tested = len(ticker_df)

                breadth_summary = pd.DataFrame([{
                    "Tickers Tested": tested,
                    "Positive Excess Tickers": positive_excess,
                    "Positive Excess Breadth %": round(positive_excess / tested * 100, 2),
                    "Positive Return Tickers": positive_return,
                    "Positive Return Breadth %": round(positive_return / tested * 100, 2),
                    "Median Top20 Avg 20D %": round(ticker_df["Top20 Avg 20D %"].median(), 2),
                    "Median Top20 Excess %": round(ticker_df["Top20 Excess %"].median(), 2),
                }])

                st.markdown("**E. Bullseye 4.0 ticker-breadth summary — broad universe**")
                st.dataframe(breadth_summary, use_container_width=True, hide_index=True)

                with st.expander("Ticker-by-ticker broad-universe results"):
                    st.dataframe(
                        ticker_df.sort_values("Top20 Excess %", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )

            st.download_button(
                "Download Phase 4C broad-universe CSV",
                broad.to_csv(index=False),
                "bullseye_phase4c_broad_universe.csv",
                "text/csv",
            )
        else:
            st.warning("No Phase 4C broad-universe samples were returned.")


if run_phase4d:
    with st.spinner("Running Phase 4D high-conviction threshold validation..."):
        broad_tickers = sorted(set(BROAD_TICKERS))
        tickers2 = sorted(set(broad_tickers + ["SPY"]))
        data = download_prices(tickers2)
        spy = one_symbol(data, "SPY")
        threshold_rows = []

        if spy is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in broad_tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                threshold_rows.extend(
                    point_in_time_backtest_symbol(
                        df,
                        spy,
                        t,
                        lookback_days=1260,
                        step=20,
                    )
                )

        if threshold_rows:
            td = pd.DataFrame(threshold_rows).dropna(
                subset=["Bullseye 4.0 Score", "20D Forward %", "Date"]
            ).copy()
            td["Date"] = pd.to_datetime(td["Date"])

            st.subheader("🎯 Phase 4D High-Conviction Threshold Validation")
            st.caption(
                "Bullseye 4.0 is frozen. This test measures where score thresholds begin to produce "
                "meaningfully stronger 1–4 week outcomes without collapsing the sample size."
            )

            thresholds = [70, 80, 85, 90, 92.5, 95]

            # A) Full-sample threshold ladder
            ladder_rows = []
            for threshold in thresholds:
                subset = td[td["Bullseye 4.0 Score"] >= threshold].copy()
                if len(subset) < 10:
                    continue
                ladder_rows.append({
                    "Threshold": f"{threshold}+",
                    "Samples": len(subset),
                    "Tickers": subset["Ticker"].nunique(),
                    "Avg 5D %": round(subset["5D Forward %"].mean(), 2),
                    "Avg 10D %": round(subset["10D Forward %"].mean(), 2),
                    "Avg 15D %": round(subset["15D Forward %"].mean(), 2),
                    "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                    "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                    "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                    "20D Hit 10% %": round((subset["20D Forward %"] >= 10).mean() * 100, 2),
                })

            ladder_df = pd.DataFrame(ladder_rows)
            st.markdown("**A. Bullseye 4.0 threshold ladder**")
            st.dataframe(ladder_df, use_container_width=True, hide_index=True)

            # B) Separate historical periods
            unique_dates = sorted(td["Date"].unique())
            cuts = np.array_split(np.array(unique_dates), 3)
            period_names = ["Older period", "Middle period", "Recent period"]
            period_rows = []

            for period_name, dates in zip(period_names, cuts):
                if len(dates) == 0:
                    continue
                start_date = pd.Timestamp(dates[0])
                end_date = pd.Timestamp(dates[-1])
                block = td[(td["Date"] >= start_date) & (td["Date"] <= end_date)].copy()

                for threshold in thresholds:
                    subset = block[block["Bullseye 4.0 Score"] >= threshold].copy()
                    if len(subset) < 5:
                        continue
                    period_rows.append({
                        "Period": period_name,
                        "Threshold": f"{threshold}+",
                        "Samples": len(subset),
                        "Tickers": subset["Ticker"].nunique(),
                        "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                        "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                        "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                        "20D Hit 10% %": round((subset["20D Forward %"] >= 10).mean() * 100, 2),
                    })

            period_df = pd.DataFrame(period_rows)
            st.markdown("**B. Threshold performance by historical period**")
            st.dataframe(period_df, use_container_width=True, hide_index=True)

            # C) Robustness scorecard by threshold
            scorecard_rows = []
            for threshold in thresholds:
                temp = period_df[period_df["Threshold"] == f"{threshold}+"].copy()
                if len(temp) == 0:
                    continue
                scorecard_rows.append({
                    "Threshold": f"{threshold}+",
                    "Periods Tested": temp["Period"].nunique(),
                    "Avg Period 20D %": round(temp["Avg 20D %"].mean(), 2),
                    "Worst Period 20D %": round(temp["Avg 20D %"].min(), 2),
                    "Best Period 20D %": round(temp["Avg 20D %"].max(), 2),
                    "Positive Periods": int((temp["Avg 20D %"] > 0).sum()),
                    "Avg Win %": round(temp["20D Win %"].mean(), 2),
                    "Avg Hit 5%": round(temp["20D Hit 5% %"].mean(), 2),
                    "Avg Hit 10%": round(temp["20D Hit 10% %"].mean(), 2),
                })

            scorecard = pd.DataFrame(scorecard_rows)
            if len(scorecard):
                scorecard = scorecard.sort_values(
                    ["Avg Period 20D %", "Worst Period 20D %"],
                    ascending=[False, False],
                )
                st.markdown("**C. Threshold robustness scorecard**")
                st.dataframe(scorecard, use_container_width=True, hide_index=True)

            # D) Ticker breadth by threshold
            breadth_rows = []
            for threshold in thresholds:
                subset = td[td["Bullseye 4.0 Score"] >= threshold].copy()
                if len(subset) < 10:
                    continue

                ticker_stats = (
                    subset.groupby("Ticker", observed=True)
                    .agg(
                        Samples=("Ticker", "count"),
                        Avg_20D=("20D Forward %", "mean"),
                        Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                        Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                    )
                    .reset_index()
                )

                ticker_stats = ticker_stats[ticker_stats["Samples"] >= 2].copy()
                if len(ticker_stats) == 0:
                    continue

                positive_return = int((ticker_stats["Avg_20D"] > 0).sum())
                breadth_rows.append({
                    "Threshold": f"{threshold}+",
                    "Tickers Tested": len(ticker_stats),
                    "Positive Return Tickers": positive_return,
                    "Positive Return Breadth %": round(
                        positive_return / len(ticker_stats) * 100, 2
                    ),
                    "Median Ticker Avg 20D %": round(ticker_stats["Avg_20D"].median(), 2),
                    "Median Ticker Win %": round(ticker_stats["Win_20D"].median(), 2),
                    "Median Ticker Hit 5%": round(ticker_stats["Hit_5pct_20D"].median(), 2),
                })

            breadth_df = pd.DataFrame(breadth_rows)
            st.markdown("**D. Threshold ticker-breadth summary**")
            st.dataframe(breadth_df, use_container_width=True, hide_index=True)

            # E) Convenience comparison around the likely decision zone
            focus_thresholds = ["85+", "90+", "92.5+", "95+"]
            focus = ladder_df[ladder_df["Threshold"].isin(focus_thresholds)].copy()
            st.markdown("**E. High-conviction decision zone**")
            st.dataframe(focus, use_container_width=True, hide_index=True)

            st.download_button(
                "Download Phase 4D threshold CSV",
                td.to_csv(index=False),
                "bullseye_phase4d_thresholds.csv",
                "text/csv",
            )
        else:
            st.warning("No Phase 4D threshold samples were returned.")


if run_phase4e:
    with st.spinner("Running Phase 4E high-score diagnostics..."):
        broad_tickers = sorted(set(BROAD_TICKERS))
        tickers2 = sorted(set(broad_tickers + ["SPY"]))
        data = download_prices(tickers2)
        spy = one_symbol(data, "SPY")
        diag_rows = []

        if spy is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in broad_tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                diag_rows.extend(
                    point_in_time_backtest_symbol(
                        df,
                        spy,
                        t,
                        lookback_days=1260,
                        step=20,
                    )
                )

        if diag_rows:
            de = pd.DataFrame(diag_rows).dropna(
                subset=["Bullseye 4.0 Score", "20D Forward %", "Date"]
            ).copy()
            de["Date"] = pd.to_datetime(de["Date"])

            st.subheader("🔎 Phase 4E High-Conviction Diagnostics")
            st.caption(
                "Bullseye 4.0 remains frozen. This test asks what separates winners from failures "
                "inside already-high Bullseye scores."
            )

            thresholds = [90, 92.5, 95]
            compare_rows = []

            feature_cols = [
                "Experimental 3.0 Score",
                "4.0 Accelerator",
                "Beta vs SPY",
                "Ann Vol %",
                "Avg $ Volume 60D ($M)",
                "20D Return %",
                "60D Return %",
                "120D Return %",
                "Relative Strength",
                "RSI",
                "Momentum",
                "Volume",
                "Technical",
                "Extension Penalty",
                "Dist 20MA %",
                "Dist 50MA %",
                "Market Regime",
            ]

            for threshold in thresholds:
                subset = de[de["Bullseye 4.0 Score"] >= threshold].copy()
                if len(subset) < 10:
                    continue

                subset["Outcome"] = np.where(
                    subset["20D Forward %"] >= 5,
                    "Hit +5%",
                    "Below +5%",
                )

                grouped = subset.groupby("Outcome", observed=True)[feature_cols].mean().round(2)
                grouped["Samples"] = subset.groupby("Outcome", observed=True).size()
                grouped = grouped.reset_index()
                grouped.insert(0, "Threshold", f"{threshold}+")
                compare_rows.append(grouped)

            if compare_rows:
                compare_df = pd.concat(compare_rows, ignore_index=True)
                st.markdown("**A. Winner vs below-5% characteristics by threshold**")
                st.dataframe(compare_df, use_container_width=True, hide_index=True)

            top95 = de[de["Bullseye 4.0 Score"] >= 95].copy()
            if len(top95) >= 20:
                top95["Winner"] = top95["20D Forward %"] >= 5
                feature_diff_rows = []

                for col in feature_cols:
                    winners = top95.loc[top95["Winner"], col].dropna()
                    losers = top95.loc[~top95["Winner"], col].dropna()
                    if len(winners) < 5 or len(losers) < 5:
                        continue
                    feature_diff_rows.append({
                        "Feature": col,
                        "Winner Avg": round(winners.mean(), 2),
                        "Below +5% Avg": round(losers.mean(), 2),
                        "Difference": round(winners.mean() - losers.mean(), 2),
                    })

                feature_diff = pd.DataFrame(feature_diff_rows)
                if len(feature_diff):
                    feature_diff["Abs Difference"] = feature_diff["Difference"].abs()
                    feature_diff = feature_diff.sort_values("Abs Difference", ascending=False)
                    st.markdown("**B. 95+ feature differences: +5% winners vs non-winners**")
                    st.dataframe(
                        feature_diff.drop(columns=["Abs Difference"]),
                        use_container_width=True,
                        hide_index=True,
                    )

                range_rows = []
                range_specs = [
                    ("RSI", [0, 60, 70, 80, 100], ["<60", "60–69.9", "70–79.9", "80+"]),
                    ("Beta vs SPY", [-100, 1.0, 1.5, 2.0, 100], ["<1.0", "1.0–1.49", "1.5–1.99", "2.0+"]),
                    ("Ann Vol %", [0, 30, 45, 60, 1000], ["<30", "30–44.9", "45–59.9", "60+"]),
                    ("Dist 20MA %", [-1000, 5, 10, 15, 1000], ["<5%", "5–9.9%", "10–14.9%", "15%+"]),
                ]

                for col, bins, labels in range_specs:
                    temp = top95[[col, "20D Forward %"]].dropna().copy()
                    if len(temp) < 10:
                        continue
                    temp["Range"] = pd.cut(temp[col], bins=bins, labels=labels, include_lowest=True)
                    grouped = (
                        temp.groupby("Range", observed=True)
                        .agg(
                            Samples=(col, "count"),
                            Avg_20D=("20D Forward %", "mean"),
                            Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                            Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                            Hit_10pct_20D=("20D Forward %", lambda x: (x >= 10).mean() * 100),
                        )
                        .reset_index()
                    )
                    for _, r in grouped.iterrows():
                        range_rows.append({
                            "Feature": col,
                            "Range": r["Range"],
                            "Samples": int(r["Samples"]),
                            "Avg 20D %": round(r["Avg_20D"], 2),
                            "20D Win %": round(r["Win_20D"], 2),
                            "20D Hit 5% %": round(r["Hit_5pct_20D"], 2),
                            "20D Hit 10% %": round(r["Hit_10pct_20D"], 2),
                        })

                if range_rows:
                    st.markdown("**C. 95+ range studies**")
                    st.dataframe(pd.DataFrame(range_rows), use_container_width=True, hide_index=True)

                top95["Regime Group"] = np.where(
                    top95["Market Regime"] >= 7,
                    "Stronger market",
                    "Weaker market",
                )

                regime_df = (
                    top95.groupby("Regime Group", observed=True)
                    .agg(
                        Samples=("Ticker", "count"),
                        Avg_5D=("5D Forward %", "mean"),
                        Avg_10D=("10D Forward %", "mean"),
                        Avg_20D=("20D Forward %", "mean"),
                        Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                        Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                        Hit_10pct_20D=("20D Forward %", lambda x: (x >= 10).mean() * 100),
                    )
                    .reset_index()
                )
                for col in ["Avg_5D", "Avg_10D", "Avg_20D", "Win_20D", "Hit_5pct_20D", "Hit_10pct_20D"]:
                    regime_df[col] = regime_df[col].round(2)

                st.markdown("**D. 95+ setups by market regime**")
                st.dataframe(regime_df, use_container_width=True, hide_index=True)

                ticker_df = (
                    top95.groupby("Ticker", observed=True)
                    .agg(
                        Samples=("Ticker", "count"),
                        Avg_20D=("20D Forward %", "mean"),
                        Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                        Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                    )
                    .reset_index()
                )
                ticker_df = ticker_df[ticker_df["Samples"] >= 2].copy()

                if len(ticker_df):
                    breadth = pd.DataFrame([{
                        "Tickers Tested": len(ticker_df),
                        "Positive Return Tickers": int((ticker_df["Avg_20D"] > 0).sum()),
                        "Positive Return Breadth %": round(
                            (ticker_df["Avg_20D"] > 0).mean() * 100, 2
                        ),
                        "Median Ticker Avg 20D %": round(ticker_df["Avg_20D"].median(), 2),
                        "Median Ticker Hit 5%": round(ticker_df["Hit_5pct_20D"].median(), 2),
                    }])

                    st.markdown("**E. 95+ ticker-breadth summary**")
                    st.dataframe(breadth, use_container_width=True, hide_index=True)

            st.download_button(
                "Download Phase 4E diagnostics CSV",
                de.to_csv(index=False),
                "bullseye_phase4e_high_score_diagnostics.csv",
                "text/csv",
            )
        else:
            st.warning("No Phase 4E diagnostic samples were returned.")



if run_phase4f:
    with st.spinner("Running Phase 4F confirmation-layer test..."):
        broad_tickers = sorted(set(BROAD_TICKERS))
        tickers2 = sorted(set(broad_tickers + ["SPY"]))
        data = download_prices(tickers2)
        spy = one_symbol(data, "SPY")
        rows4f = []
        if spy is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in broad_tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                rows4f.extend(point_in_time_backtest_symbol(df, spy, t, lookback_days=1260, step=20))

        if rows4f:
            f4 = pd.DataFrame(rows4f).dropna(subset=["Bullseye 4.0 Score", "20D Forward %", "Date"]).copy()
            f4["Date"] = pd.to_datetime(f4["Date"])
            base = f4[f4["Bullseye 4.0 Score"] >= 95].copy()
            st.subheader("🧭 Phase 4F Confirmation-Layer Test")
            st.caption("Bullseye 4.0 is unchanged. Phase 4F tests whether liquidity, long-term trend, accelerator strength, beta, volatility, RSI and extension can improve selection inside the 95+ zone.")

            if len(base) >= 20:
                # Fixed, interpretable confirmation rules derived from 4E.
                base["Liquid"] = base["Avg $ Volume 60D ($M)"] >= 5000
                base["Trend60"] = base["60D Return %"] >= 50
                base["Trend120"] = base["120D Return %"] >= 70
                base["Accel10"] = base["4.0 Accelerator"] >= 10
                base["BetaSweet"] = base["Beta vs SPY"].between(1.5, 2.0, inclusive="left")
                base["VolSweet"] = base["Ann Vol %"].between(30, 45, inclusive="left")
                base["RSI80"] = base["RSI"] >= 80
                base["NotExtreme20MA"] = base["Dist 20MA %"] < 20
                base["CoreConfirm"] = base[["Liquid", "Trend60", "Trend120", "Accel10"]].sum(axis=1)
                base["BroadConfirm"] = base[["Liquid", "Trend60", "Trend120", "Accel10", "BetaSweet", "VolSweet", "RSI80", "NotExtreme20MA"]].sum(axis=1)

                rules = [
                    ("95+ baseline", pd.Series(True, index=base.index)),
                    ("Liquidity >= $5B", base["Liquid"]),
                    ("120D trend >= 70%", base["Trend120"]),
                    ("Accelerator >= 10", base["Accel10"]),
                    ("Core confirmation >= 2 of 4", base["CoreConfirm"] >= 2),
                    ("Core confirmation >= 3 of 4", base["CoreConfirm"] >= 3),
                    ("Core confirmation = 4 of 4", base["CoreConfirm"] >= 4),
                    ("Broad confirmation >= 5 of 8", base["BroadConfirm"] >= 5),
                    ("Broad confirmation >= 6 of 8", base["BroadConfirm"] >= 6),
                ]

                def metrics(name, mask, block=base):
                    x = block.loc[mask].copy()
                    if len(x) == 0:
                        return None
                    return {
                        "Confirmation Rule": name,
                        "Samples": len(x),
                        "Tickers": x["Ticker"].nunique(),
                        "Avg 5D %": round(x["5D Forward %"].mean(), 2),
                        "Avg 10D %": round(x["10D Forward %"].mean(), 2),
                        "Avg 20D %": round(x["20D Forward %"].mean(), 2),
                        "20D Win %": round((x["20D Forward %"] > 0).mean()*100, 2),
                        "20D Hit 5% %": round((x["20D Forward %"] >= 5).mean()*100, 2),
                        "20D Hit 10% %": round((x["20D Forward %"] >= 10).mean()*100, 2),
                    }

                head = [metrics(n,m) for n,m in rules]
                head = pd.DataFrame([r for r in head if r is not None])
                st.markdown("**A. 95+ confirmation-rule head-to-head**")
                st.dataframe(head, use_container_width=True, hide_index=True)

                # Confirmation-count ladders reveal whether quality rises monotonically.
                ladder=[]
                for score_col, max_score in [("CoreConfirm",4),("BroadConfirm",8)]:
                    for k in range(0,max_score+1):
                        mask=base[score_col] >= k
                        r=metrics(f"{score_col} >= {k}", mask)
                        if r: ladder.append(r)
                st.markdown("**B. Confirmation-count ladder**")
                st.dataframe(pd.DataFrame(ladder), use_container_width=True, hide_index=True)

                # Historical-period robustness for the most practical filters.
                q1,q2 = f4["Date"].quantile([1/3,2/3])
                periods=[("Older period", f4["Date"]<=q1),("Middle period",(f4["Date"]>q1)&(f4["Date"]<=q2)),("Recent period",f4["Date"]>q2)]
                period_rows=[]
                for pname, pmask in periods:
                    pb=base.loc[pmask.reindex(base.index, fill_value=False)].copy()
                    if len(pb)==0: continue
                    prules=[("95+ baseline",pd.Series(True,index=pb.index)),("Core >=2",pb["CoreConfirm"]>=2),("Core >=3",pb["CoreConfirm"]>=3),("Broad >=5",pb["BroadConfirm"]>=5),("Broad >=6",pb["BroadConfirm"]>=6)]
                    for rn,rm in prules:
                        r=metrics(rn,rm,pb)
                        if r:
                            r={"Period":pname,**r}
                            period_rows.append(r)
                st.markdown("**C. Confirmation rules by historical period**")
                st.dataframe(pd.DataFrame(period_rows), use_container_width=True, hide_index=True)

                # Market-regime check for practical confirmation variants.
                base["Regime Group"] = np.where(base["Market Regime"] >= 7,"Stronger market","Weaker market")
                regime_rows=[]
                for rg, rb in base.groupby("Regime Group", observed=True):
                    for rn,rm in [("95+ baseline",pd.Series(True,index=rb.index)),("Core >=2",rb["CoreConfirm"]>=2),("Core >=3",rb["CoreConfirm"]>=3),("Broad >=5",rb["BroadConfirm"]>=5)]:
                        r=metrics(rn,rm,rb)
                        if r: regime_rows.append({"Market Regime":rg,**r})
                st.markdown("**D. Confirmation rules by market regime**")
                st.dataframe(pd.DataFrame(regime_rows), use_container_width=True, hide_index=True)

                # Ticker breadth for the candidate practical rule Core >=2.
                cand=base[base["CoreConfirm"]>=2].copy()
                if len(cand):
                    tb=(cand.groupby("Ticker",observed=True).agg(Samples=("Ticker","count"),Avg_20D=("20D Forward %","mean"),Hit_5=("20D Forward %",lambda x:(x>=5).mean()*100)).reset_index())
                    tb=tb[tb["Samples"]>=2]
                    if len(tb):
                        breadth=pd.DataFrame([{
                            "Rule":"95+ and Core confirmation >=2",
                            "Tickers Tested":len(tb),
                            "Positive Return Tickers":int((tb["Avg_20D"]>0).sum()),
                            "Positive Return Breadth %":round((tb["Avg_20D"]>0).mean()*100,2),
                            "Median Ticker Avg 20D %":round(tb["Avg_20D"].median(),2),
                            "Median Ticker Hit 5%":round(tb["Hit_5"].median(),2),
                        }])
                        st.markdown("**E. Candidate confirmation-layer ticker breadth**")
                        st.dataframe(breadth,use_container_width=True,hide_index=True)

                st.download_button("Download Phase 4F confirmation CSV", base.to_csv(index=False), "bullseye_phase4f_confirmation_layer.csv", "text/csv")
            else:
                st.warning("Not enough 95+ samples were returned for Phase 4F.")
        else:
            st.warning("No Phase 4F samples were returned.")


if run_phase4g:
    with st.spinner("Running Phase 4G confirmation robustness validation..."):
        broad_tickers = sorted(set(BROAD_TICKERS))
        tickers2 = sorted(set(broad_tickers + ["SPY"]))
        data = download_prices(tickers2)
        spy = one_symbol(data, "SPY")
        rows_4g = []

        if spy is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in broad_tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                rows_4g.extend(
                    point_in_time_backtest_symbol(
                        df,
                        spy,
                        t,
                        lookback_days=1260,
                        step=20,
                    )
                )

        if rows_4g:
            g = pd.DataFrame(rows_4g).dropna(
                subset=["Bullseye 4.0 Score", "20D Forward %", "Date"]
            ).copy()
            g["Date"] = pd.to_datetime(g["Date"])

            # Freeze the exact 4F rules.
            g["Rule 95+ baseline"] = g["Bullseye 4.0 Score"] >= 95
            g["Rule Accelerator>=10"] = (
                (g["Bullseye 4.0 Score"] >= 95)
                & (g["4.0 Accelerator"] >= 10)
            )

            # Rebuild Core confirmation exactly as used in 4F:
            # liquidity >= $5B, 120D trend >= 70%, accelerator >=10, beta >=1.5
            g["Core Count"] = (
                (g["Avg $ Volume 60D ($M)"] >= 5000).astype(int)
                + (g["120D Return %"] >= 70).astype(int)
                + (g["4.0 Accelerator"] >= 10).astype(int)
                + (g["Beta vs SPY"] >= 1.5).astype(int)
            )

            # Rebuild Broad confirmation exactly as used in 4F:
            # core four + RSI>=70 + vol 30-60 + dist20 10-20 + 60D return>=40
            g["Broad Count"] = (
                g["Core Count"]
                + (g["RSI"] >= 70).astype(int)
                + ((g["Ann Vol %"] >= 30) & (g["Ann Vol %"] <= 60)).astype(int)
                + ((g["Dist 20MA %"] >= 10) & (g["Dist 20MA %"] <= 20)).astype(int)
                + (g["60D Return %"] >= 40).astype(int)
            )

            g["Rule Core>=3"] = (
                (g["Bullseye 4.0 Score"] >= 95)
                & (g["Core Count"] >= 3)
            )
            g["Rule Broad>=5"] = (
                (g["Bullseye 4.0 Score"] >= 95)
                & (g["Broad Count"] >= 5)
            )
            g["Rule Liquidity>=$5B"] = (
                (g["Bullseye 4.0 Score"] >= 95)
                & (g["Avg $ Volume 60D ($M)"] >= 5000)
            )

            st.subheader("🧱 Phase 4G Confirmation Robustness")
            st.caption(
                "All rules are frozen exactly from Phase 4F. No thresholds are being adjusted in this test."
            )

            rule_map = [
                ("95+ baseline", "Rule 95+ baseline"),
                ("Accelerator >=10", "Rule Accelerator>=10"),
                ("Core >=3 of 4", "Rule Core>=3"),
                ("Broad >=5 of 8", "Rule Broad>=5"),
                ("Liquidity >=$5B", "Rule Liquidity>=$5B"),
            ]

            # A) Full-sample frozen-rule comparison
            comp_rows = []
            for label, col in rule_map:
                subset = g[g[col]].copy()
                if len(subset) < 5:
                    continue
                comp_rows.append({
                    "Rule": label,
                    "Samples": len(subset),
                    "Tickers": subset["Ticker"].nunique(),
                    "Avg 5D %": round(subset["5D Forward %"].mean(), 2),
                    "Avg 10D %": round(subset["10D Forward %"].mean(), 2),
                    "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                    "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                    "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                    "20D Hit 10% %": round((subset["20D Forward %"] >= 10).mean() * 100, 2),
                })

            comp_df = pd.DataFrame(comp_rows)
            st.markdown("**A. Frozen-rule full-sample comparison**")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # B) Separate chronological blocks
            unique_dates = sorted(g["Date"].unique())
            cuts = np.array_split(np.array(unique_dates), 3)
            period_names = ["Older period", "Middle period", "Recent period"]

            period_rows = []
            for period_name, dates in zip(period_names, cuts):
                if len(dates) == 0:
                    continue
                start_date = pd.Timestamp(dates[0])
                end_date = pd.Timestamp(dates[-1])
                block = g[(g["Date"] >= start_date) & (g["Date"] <= end_date)].copy()

                for label, col in rule_map:
                    subset = block[block[col]].copy()
                    if len(subset) < 3:
                        continue
                    period_rows.append({
                        "Period": period_name,
                        "Start": start_date.date(),
                        "End": end_date.date(),
                        "Rule": label,
                        "Samples": len(subset),
                        "Tickers": subset["Ticker"].nunique(),
                        "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                        "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                        "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                        "20D Hit 10% %": round((subset["20D Forward %"] >= 10).mean() * 100, 2),
                    })

            period_df = pd.DataFrame(period_rows)
            st.markdown("**B. Frozen rules by separate historical period**")
            st.dataframe(period_df, use_container_width=True, hide_index=True)

            # C) Robustness scorecard
            scorecard_rows = []
            for label, _ in rule_map:
                temp = period_df[period_df["Rule"] == label].copy()
                if len(temp) == 0:
                    continue
                scorecard_rows.append({
                    "Rule": label,
                    "Periods Tested": temp["Period"].nunique(),
                    "Avg Period 20D %": round(temp["Avg 20D %"].mean(), 2),
                    "Worst Period 20D %": round(temp["Avg 20D %"].min(), 2),
                    "Best Period 20D %": round(temp["Avg 20D %"].max(), 2),
                    "Positive Periods": int((temp["Avg 20D %"] > 0).sum()),
                    "Avg Win %": round(temp["20D Win %"].mean(), 2),
                    "Avg Hit 5%": round(temp["20D Hit 5% %"].mean(), 2),
                    "Avg Hit 10%": round(temp["20D Hit 10% %"].mean(), 2),
                })

            scorecard_df = pd.DataFrame(scorecard_rows)
            if len(scorecard_df):
                scorecard_df = scorecard_df.sort_values(
                    ["Avg Period 20D %", "Worst Period 20D %"],
                    ascending=[False, False],
                )
                st.markdown("**C. Frozen-rule robustness scorecard**")
                st.dataframe(scorecard_df, use_container_width=True, hide_index=True)

            # D) Ticker breadth
            breadth_rows = []
            for label, col in rule_map:
                subset = g[g[col]].copy()
                if len(subset) < 5:
                    continue

                ticker_stats = (
                    subset.groupby("Ticker", observed=True)
                    .agg(
                        Samples=("Ticker", "count"),
                        Avg_20D=("20D Forward %", "mean"),
                        Win_20D=("20D Forward %", lambda x: (x > 0).mean() * 100),
                        Hit_5pct_20D=("20D Forward %", lambda x: (x >= 5).mean() * 100),
                    )
                    .reset_index()
                )
                ticker_stats = ticker_stats[ticker_stats["Samples"] >= 2].copy()
                if len(ticker_stats) == 0:
                    continue

                positive = int((ticker_stats["Avg_20D"] > 0).sum())
                breadth_rows.append({
                    "Rule": label,
                    "Tickers Tested": len(ticker_stats),
                    "Positive Return Tickers": positive,
                    "Positive Return Breadth %": round(
                        positive / len(ticker_stats) * 100, 2
                    ),
                    "Median Ticker Avg 20D %": round(ticker_stats["Avg_20D"].median(), 2),
                    "Median Ticker Win %": round(ticker_stats["Win_20D"].median(), 2),
                    "Median Ticker Hit 5%": round(ticker_stats["Hit_5pct_20D"].median(), 2),
                })

            breadth_df = pd.DataFrame(breadth_rows)
            st.markdown("**D. Frozen-rule ticker breadth**")
            st.dataframe(breadth_df, use_container_width=True, hide_index=True)

            # E) Stronger vs weaker market
            g["Regime Group"] = np.where(
                g["Market Regime"] >= 7,
                "Stronger market",
                "Weaker market",
            )
            regime_rows = []
            for regime_name, block in g.groupby("Regime Group", observed=True):
                for label, col in rule_map:
                    subset = block[block[col]].copy()
                    if len(subset) < 3:
                        continue
                    regime_rows.append({
                        "Market Regime": regime_name,
                        "Rule": label,
                        "Samples": len(subset),
                        "Tickers": subset["Ticker"].nunique(),
                        "Avg 20D %": round(subset["20D Forward %"].mean(), 2),
                        "20D Win %": round((subset["20D Forward %"] > 0).mean() * 100, 2),
                        "20D Hit 5% %": round((subset["20D Forward %"] >= 5).mean() * 100, 2),
                        "20D Hit 10% %": round((subset["20D Forward %"] >= 10).mean() * 100, 2),
                    })

            regime_df = pd.DataFrame(regime_rows)
            st.markdown("**E. Frozen rules by market regime**")
            st.dataframe(regime_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download Phase 4G robustness CSV",
                g.to_csv(index=False),
                "bullseye_phase4g_confirmation_robustness.csv",
                "text/csv",
            )
        else:
            st.warning("No Phase 4G robustness samples were returned.")


if run_phase4h:
    with st.spinner("Running Phase 4H signal-architecture validation..."):
        broad_tickers = sorted(set(BROAD_TICKERS))
        data = download_prices(sorted(set(broad_tickers + ["SPY"])))
        spy = one_symbol(data, "SPY")
        rows_4h = []
        if spy is not None:
            for t in broad_tickers:
                df = one_symbol(data, t)
                if df is not None:
                    rows_4h.extend(point_in_time_backtest_symbol(df, spy, t, lookback_days=1260, step=20))
        if rows_4h:
            h = pd.DataFrame(rows_4h).copy()
            h["4H Core Count"] = (
                (h["Avg $ Volume 60D ($M)"] >= 5000).astype(int)
                + (h["120D Return %"] >= 70).astype(int)
                + (h["4.0 Accelerator"] >= 10).astype(int)
                + (h["Beta vs SPY"] >= 1.5).astype(int)
            )
            def _tier(r):
                s, c, a = r["Bullseye 4.0 Score"], r["4H Core Count"], r["4.0 Accelerator"]
                if s >= 95 and c >= 3: return "Elite Confirmed"
                if s >= 95 and a >= 10: return "Confirmed Prime"
                if s >= 95: return "Prime 95+"
                if s >= 92.5: return "Very High Conviction"
                if s >= 90: return "High Conviction"
                return "Standard"
            h["4H Signal Tier"] = h.apply(_tier, axis=1)
            st.subheader("🏗️ Phase 4H Final Signal Architecture")
            summary = (
                h.groupby("4H Signal Tier", observed=True)
                .agg(
                    Samples=("Ticker","count"),
                    Tickers=("Ticker","nunique"),
                    Avg_20D=("20D Forward %","mean"),
                    Win_20D=("20D Forward %",lambda x:(x>0).mean()*100),
                    Hit_5=("20D Forward %",lambda x:(x>=5).mean()*100),
                    Hit_10=("20D Forward %",lambda x:(x>=10).mean()*100),
                ).reset_index()
            )
            for c in ["Avg_20D","Win_20D","Hit_5","Hit_10"]:
                summary[c] = summary[c].round(2)
            st.dataframe(summary, use_container_width=True, hide_index=True)

if run_phase4i:
    with st.spinner("Running Phase 4I live decision-screen validation..."):
        broad_tickers = sorted(set(BROAD_TICKERS))
        data = download_prices(sorted(set(broad_tickers + ["SPY"])))
        spy = one_symbol(data, "SPY")
        rows_i = []
        if spy is not None:
            for t in broad_tickers:
                df = one_symbol(data, t)
                if df is not None:
                    try:
                        r = score_stock(df, spy)
                        r["Ticker"] = t
                        rows_i.append(r)
                    except Exception:
                        continue
        if rows_i:
            d = pd.DataFrame(rows_i).sort_values(
                ["4I Action Rank","Bullseye 4.0 Score","4H Core Count","4.0 Accelerator"],
                ascending=[False,False,False,False],
            )
            st.subheader("🖥️ Phase 4I Live Decision Screen")
            hi = d[d["4I Action Rank"] >= 1]
            st.dataframe(
                hi[["Ticker","4I Action","4H Signal Tier","Bullseye 4.0 Score",
                    "4H Signal Badges","4I Why","Price","4H Core Count",
                    "4.0 Accelerator","Beta 120D","Avg $ Volume 60D ($M)",
                    "120D %","RSI","Dist 20MA %","Market Regime"]],
                use_container_width=True, hide_index=True
            )

if run_phase4j:
    with st.spinner("Building Phase 4J forward signal journal..."):
        journal_tickers = sorted(set(BROAD_TICKERS))
        data = download_prices(sorted(set(journal_tickers + ["SPY"])))
        spy = one_symbol(data, "SPY")
        journal_rows = []
        if spy is not None:
            for t in journal_tickers:
                df = one_symbol(data, t)
                if df is None:
                    continue
                try:
                    row = score_stock(df, spy)
                    row["Ticker"] = t
                    journal_rows.append(row)
                except Exception:
                    continue

        if journal_rows:
            j = pd.DataFrame(journal_rows).copy()
            now_ts = pd.Timestamp.now()
            j["Signal Timestamp"] = now_ts.strftime("%Y-%m-%d %H:%M:%S")
            j["Signal Date"] = now_ts.strftime("%Y-%m-%d")
            j["Entry Price"] = j["Price"]
            j["Frozen Model"] = "Bullseye 4.0 + Phase 4I decision architecture"
            for days in (5, 10, 20):
                j[f"{days}D Review Date"] = ""
                j[f"{days}D Review Price"] = np.nan
                j[f"{days}D Return %"] = np.nan
                j[f"{days}D Reviewed"] = False

            forward = j[j["4I Action Rank"] >= 1].copy().sort_values(
                ["4I Action Rank","Bullseye 4.0 Score","4H Core Count","4.0 Accelerator"],
                ascending=[False,False,False,False],
            )
            st.subheader("📝 Phase 4J Forward Signal Journal")
            st.caption("Save this CSV unchanged. Phase 4K can update it when 5, 10, and 20 trading-day checkpoints become available.")
            if len(forward):
                st.dataframe(
                    forward[["Signal Timestamp","Ticker","4I Action","4H Signal Tier",
                             "Bullseye 4.0 Score","Entry Price","4H Core Count",
                             "4.0 Accelerator","4H Signal Badges","Beta 120D",
                             "Avg $ Volume 60D ($M)","120D %","RSI",
                             "Dist 20MA %","Market Regime"]],
                    use_container_width=True, hide_index=True
                )
                journal_cols = [
                    "Signal Timestamp","Signal Date","Ticker","Entry Price",
                    "4I Action","4I Action Rank","4H Signal Tier","Bullseye 4.0 Score",
                    "4H Core Count","4.0 Accelerator","4H Signal Badges","Beta 120D",
                    "Avg $ Volume 60D ($M)","120D %","RSI","Dist 20MA %","Market Regime",
                    "5D Review Date","5D Review Price","5D Return %","5D Reviewed",
                    "10D Review Date","10D Review Price","10D Return %","10D Reviewed",
                    "20D Review Date","20D Review Price","20D Return %","20D Reviewed",
                    "Frozen Model",
                ]
                st.download_button(
                    "Download today's Phase 4J forward-signal journal",
                    forward[journal_cols].to_csv(index=False),
                    f"bullseye_phase4j_forward_journal_{now_ts.strftime('%Y%m%d')}.csv",
                    "text/csv",
                )
            else:
                st.warning("No 90+ Bullseye signals were found today.")

if run_phase4k:
    st.subheader("📈 Phase 4K Forward-Journal Review")
    if journal_upload is None:
        st.warning("Upload a saved Phase 4J journal CSV first.")
    else:
        try:
            journal = pd.read_csv(journal_upload)
            required = {"Signal Date","Ticker","Entry Price"}
            missing = required.difference(journal.columns)
            if missing:
                st.error("Journal is missing required columns: " + ", ".join(sorted(missing)))
            else:
                journal["Signal Date"] = pd.to_datetime(journal["Signal Date"], errors="coerce")
                journal = journal.dropna(subset=["Signal Date","Ticker","Entry Price"]).copy()
                journal["Ticker"] = journal["Ticker"].astype(str).str.upper().str.strip()

                # Ensure review columns exist, including compatibility with older 4J files.
                for days in (5,10,20):
                    for col, default in [
                        (f"{days}D Review Date",""),
                        (f"{days}D Review Price",np.nan),
                        (f"{days}D Return %",np.nan),
                        (f"{days}D Reviewed",False),
                    ]:
                        if col not in journal.columns:
                            journal[col] = default

                tickers_k = sorted(journal["Ticker"].dropna().unique().tolist())
                data_k = download_prices(tickers_k)
                today = pd.Timestamp.now().normalize()

                current_prices = []
                trading_days_elapsed = []

                for idx, row in journal.iterrows():
                    t = row["Ticker"]
                    df = one_symbol(data_k, t)
                    if df is None:
                        current_prices.append(np.nan)
                        trading_days_elapsed.append(np.nan)
                        continue

                    hist = df[df.index.normalize() > row["Signal Date"].normalize()].copy()
                    current_prices.append(round(float(df["Close"].iloc[-1]), 2))
                    trading_days_elapsed.append(len(hist))

                    entry = float(row["Entry Price"])
                    for days in (5,10,20):
                        if len(hist) >= days:
                            review_date = pd.Timestamp(hist.index[days-1])
                            review_price = float(hist["Close"].iloc[days-1])
                            journal.at[idx, f"{days}D Review Date"] = review_date.strftime("%Y-%m-%d")
                            journal.at[idx, f"{days}D Review Price"] = round(review_price, 2)
                            journal.at[idx, f"{days}D Return %"] = round(pct(review_price, entry), 2)
                            journal.at[idx, f"{days}D Reviewed"] = True

                journal["Current Price"] = current_prices
                journal["Trading Days Elapsed"] = trading_days_elapsed
                journal["Current Return %"] = (
                    (journal["Current Price"] / journal["Entry Price"] - 1) * 100
                ).round(2)
                journal["Last Review Run"] = today.strftime("%Y-%m-%d")

                st.markdown("**A. Journal review status**")
                status = pd.DataFrame([{
                    "Signals": len(journal),
                    "5D Complete": int(journal["5D Reviewed"].fillna(False).astype(bool).sum()),
                    "10D Complete": int(journal["10D Reviewed"].fillna(False).astype(bool).sum()),
                    "20D Complete": int(journal["20D Reviewed"].fillna(False).astype(bool).sum()),
                    "20D +5% Hits": int((pd.to_numeric(journal["20D Return %"], errors="coerce") >= 5).sum()),
                    "20D +10% Hits": int((pd.to_numeric(journal["20D Return %"], errors="coerce") >= 10).sum()),
                }])
                st.dataframe(status, use_container_width=True, hide_index=True)

                st.markdown("**B. Updated forward journal**")
                display_cols = [
                    "Signal Date","Ticker","4I Action","Bullseye 4.0 Score","Entry Price",
                    "Trading Days Elapsed","Position Mark Price","Current Return %",
                    "5D Return %","10D Return %","20D Return %"
                ]
                display_cols = [c for c in display_cols if c in journal.columns]
                st.dataframe(journal[display_cols], use_container_width=True, hide_index=True)

                completed = journal[pd.to_numeric(journal["20D Return %"], errors="coerce").notna()].copy()
                if len(completed):
                    st.markdown("**C. Completed 20D outcomes by signal level**")
                    group_col = "4I Action" if "4I Action" in completed.columns else "4H Signal Tier"
                    summary = (
                        completed.groupby(group_col, observed=True)
                        .agg(
                            Samples=("Ticker","count"),
                            Avg_20D=("20D Return %","mean"),
                            Win_20D=("20D Return %",lambda x:(x>0).mean()*100),
                            Hit_5=("20D Return %",lambda x:(x>=5).mean()*100),
                            Hit_10=("20D Return %",lambda x:(x>=10).mean()*100),
                        ).reset_index()
                    )
                    for c in ["Avg_20D","Win_20D","Hit_5","Hit_10"]:
                        summary[c] = summary[c].round(2)
                    st.dataframe(summary, use_container_width=True, hide_index=True)

                pending = journal[journal["20D Reviewed"].fillna(False).astype(bool) == False].copy()
                if len(pending):
                    st.markdown("**D. Signals still in forward test**")
                    pend_cols = ["Ticker","Signal Date","Trading Days Elapsed","Current Return %",
                                 "5D Reviewed","10D Reviewed","20D Reviewed"]
                    st.dataframe(pending[pend_cols], use_container_width=True, hide_index=True)

                st.download_button(
                    "Download updated Phase 4K journal",
                    journal.to_csv(index=False),
                    f"bullseye_phase4k_updated_journal_{today.strftime('%Y%m%d')}.csv",
                    "text/csv",
                )
        except Exception as exc:
            st.error(f"Could not review the journal: {exc}")


if run_phase4l:
    st.subheader("📊 Phase 4L Forward Performance Dashboard")
    if journal_upload is None:
        st.warning("Upload your latest saved Phase 4K journal CSV first.")
    else:
        try:
            dash = pd.read_csv(journal_upload)
            required = {"Signal Date", "Ticker", "Entry Price"}
            missing = required.difference(dash.columns)

            if missing:
                st.error("Journal is missing required columns: " + ", ".join(sorted(missing)))
            else:
                dash["Signal Date"] = pd.to_datetime(dash["Signal Date"], errors="coerce")
                dash = dash.dropna(subset=["Signal Date", "Ticker", "Entry Price"]).copy()
                dash["Ticker"] = dash["Ticker"].astype(str).str.upper().str.strip()

                # Normalize numeric outcome fields for journals created by earlier 4K builds.
                numeric_cols = [
                    "Bullseye 4.0 Score", "Entry Price", "Current Price", "Current Return %",
                    "Trading Days Elapsed", "5D Return %", "10D Return %", "20D Return %"
                ]
                for col in numeric_cols:
                    if col in dash.columns:
                        dash[col] = pd.to_numeric(dash[col], errors="coerce")

                for days in (5, 10, 20):
                    reviewed_col = f"{days}D Reviewed"
                    return_col = f"{days}D Return %"
                    if return_col not in dash.columns:
                        dash[return_col] = np.nan
                    if reviewed_col not in dash.columns:
                        dash[reviewed_col] = dash[return_col].notna()
                    else:
                        # CSV round-trips may store booleans as strings.
                        dash[reviewed_col] = (
                            dash[reviewed_col].astype(str).str.lower()
                            .map({"true": True, "false": False})
                            .fillna(dash[return_col].notna())
                            .astype(bool)
                        )

                total = len(dash)
                completed_5 = int(dash["5D Return %"].notna().sum())
                completed_10 = int(dash["10D Return %"].notna().sum())
                completed_20 = int(dash["20D Return %"].notna().sum())
                pending_20 = total - completed_20

                st.markdown("**A. Forward-test scoreboard**")
                scorecard = pd.DataFrame([{
                    "Signals Logged": total,
                    "5D Complete": completed_5,
                    "10D Complete": completed_10,
                    "20D Complete": completed_20,
                    "20D Pending": pending_20,
                    "20D Completion %": round((completed_20 / total * 100), 2) if total else 0.0,
                }])
                st.dataframe(scorecard, use_container_width=True, hide_index=True)

                st.markdown("**B. Realized performance by horizon**")
                horizon_rows = []
                for days in (5, 10, 20):
                    col = f"{days}D Return %"
                    vals = dash[col].dropna()
                    if len(vals):
                        horizon_rows.append({
                            "Horizon": f"{days}D",
                            "Completed Signals": len(vals),
                            "Avg Return %": round(float(vals.mean()), 2),
                            "Median Return %": round(float(vals.median()), 2),
                            "Win %": round(float((vals > 0).mean() * 100), 2),
                            "Hit +5% %": round(float((vals >= 5).mean() * 100), 2),
                            "Hit +10% %": round(float((vals >= 10).mean() * 100), 2),
                            "Best %": round(float(vals.max()), 2),
                            "Worst %": round(float(vals.min()), 2),
                        })
                if horizon_rows:
                    horizon_df = pd.DataFrame(horizon_rows)
                    st.dataframe(horizon_df, use_container_width=True, hide_index=True)
                    st.bar_chart(
                        horizon_df.set_index("Horizon")[["Avg Return %", "Median Return %"]],
                        use_container_width=True
                    )
                else:
                    st.info("No 5D, 10D, or 20D outcomes have matured yet.")

                st.markdown("**C. Performance by live decision level**")
                group_col = None
                if "4I Action" in dash.columns:
                    group_col = "4I Action"
                elif "4H Signal Tier" in dash.columns:
                    group_col = "4H Signal Tier"

                completed = dash[dash["20D Return %"].notna()].copy()
                if group_col and len(completed):
                    by_level = (
                        completed.groupby(group_col, observed=True)
                        .agg(
                            Samples=("Ticker", "count"),
                            Tickers=("Ticker", "nunique"),
                            Avg_20D=("20D Return %", "mean"),
                            Median_20D=("20D Return %", "median"),
                            Win_20D=("20D Return %", lambda x: (x > 0).mean() * 100),
                            Hit_5=("20D Return %", lambda x: (x >= 5).mean() * 100),
                            Hit_10=("20D Return %", lambda x: (x >= 10).mean() * 100),
                        )
                        .reset_index()
                    )
                    for c in ["Avg_20D", "Median_20D", "Win_20D", "Hit_5", "Hit_10"]:
                        by_level[c] = by_level[c].round(2)
                    st.dataframe(by_level, use_container_width=True, hide_index=True)
                    st.bar_chart(
                        by_level.set_index(group_col)[["Avg_20D", "Hit_5", "Hit_10"]],
                        use_container_width=True
                    )
                elif group_col:
                    st.info("Decision-level 20D performance will appear after the first signals complete 20 trading days.")
                else:
                    st.info("This journal does not contain a 4I Action or 4H Signal Tier column.")

                st.markdown("**D. Current open-signal monitor**")
                open_signals = dash[dash["20D Return %"].isna()].copy()
                if len(open_signals):
                    open_cols = [
                        "Signal Date", "Ticker", "4I Action", "4H Signal Tier",
                        "Bullseye 4.0 Score", "Entry Price", "Trading Days Elapsed",
                        "Current Price", "Current Return %", "5D Return %", "10D Return %"
                    ]
                    open_cols = [c for c in open_cols if c in open_signals.columns]
                    sort_cols = [c for c in ["4I Action Rank", "Bullseye 4.0 Score"] if c in open_signals.columns]
                    if sort_cols:
                        open_signals = open_signals.sort_values(sort_cols, ascending=[False] * len(sort_cols))
                    st.dataframe(open_signals[open_cols], use_container_width=True, hide_index=True)
                else:
                    st.success("Every signal in this journal has a completed 20D outcome.")

                st.markdown("**E. Completed 20D signal ledger**")
                if len(completed):
                    ledger_cols = [
                        "Signal Date", "Ticker", "4I Action", "4H Signal Tier",
                        "Bullseye 4.0 Score", "Entry Price", "20D Review Price",
                        "20D Return %", "4H Signal Badges"
                    ]
                    ledger_cols = [c for c in ledger_cols if c in completed.columns]
                    completed = completed.sort_values("20D Return %", ascending=False)
                    st.dataframe(completed[ledger_cols], use_container_width=True, hide_index=True)

                    st.markdown("**F. Forward-validation breadth**")
                    breadth = pd.DataFrame([{
                        "Completed 20D Signals": len(completed),
                        "Unique Tickers": completed["Ticker"].nunique(),
                        "Positive Return Signals": int((completed["20D Return %"] > 0).sum()),
                        "Positive Return Breadth %": round(float((completed["20D Return %"] > 0).mean() * 100), 2),
                        "+5% Hits": int((completed["20D Return %"] >= 5).sum()),
                        "+10% Hits": int((completed["20D Return %"] >= 10).sum()),
                    }])
                    st.dataframe(breadth, use_container_width=True, hide_index=True)
                else:
                    st.info("The completed 20D ledger will populate automatically as the journal matures.")

        except Exception as exc:
            st.error(f"Could not build the Phase 4L dashboard: {exc}")


if run_phase4m:
    with st.spinner("Building Phase 4M live Bullseye command center..."):
        live_tickers = sorted(set(BROAD_TICKERS))
        data_m = download_prices(sorted(set(live_tickers + ["SPY"])))
        spy_m = one_symbol(data_m, "SPY")
        live_rows = []

        if spy_m is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in live_tickers:
                df = one_symbol(data_m, t)
                if df is None:
                    continue
                try:
                    row = score_stock(df, spy_m)
                    row["Ticker"] = t
                    live_rows.append(row)
                except Exception:
                    continue

        if live_rows:
            live = pd.DataFrame(live_rows).copy()

            # Keep the validated 4I priority ordering frozen.
            live = live.sort_values(
                ["4I Action Rank", "Bullseye 4.0 Score", "4H Core Count", "4.0 Accelerator"],
                ascending=[False, False, False, False],
            )

            actionable = live[live["4I Action Rank"] >= 1].copy()

            st.subheader("🎛️ Phase 4M Live Bullseye Command Center")
            st.caption(
                "This screen reorganizes the frozen Bullseye 4.0 / 4H / 4I signals for daily use. "
                "It does not alter scoring or create new trade rules."
            )

            # A. Snapshot
            priority_count = int((actionable["4I Action"] == "Priority Watch").sum())
            strong_count = int((actionable["4I Action"] == "Strong Watch").sum())
            close_count = int((actionable["4I Action"] == "Watch Closely").sum())
            watch_count = int((actionable["4I Action"] == "Watch").sum())
            secondary_count = int((actionable["4I Action"] == "Secondary Watch").sum())

            st.markdown("**A. Today's signal snapshot**")
            snapshot = pd.DataFrame([{
                "Actionable Signals": len(actionable),
                "Priority Watch": priority_count,
                "Strong Watch": strong_count,
                "Watch Closely": close_count,
                "Watch": watch_count,
                "Secondary Watch": secondary_count,
                "Universe Scanned": live["Ticker"].nunique(),
            }])
            st.dataframe(snapshot, use_container_width=True, hide_index=True)

            if len(actionable):
                # B. Top board
                st.markdown("**B. Ranked live signal board**")
                board_cols = [
                    "Ticker", "4I Action", "4H Signal Tier", "Bullseye 4.0 Score",
                    "Price", "4H Signal Badges", "4I Why", "4H Core Count",
                    "4.0 Accelerator", "Beta 120D", "Avg $ Volume 60D ($M)",
                    "120D %", "RSI", "Dist 20MA %", "Rel Vol",
                    "RS vs SPY 20D", "Market Regime"
                ]
                board_cols = [c for c in board_cols if c in actionable.columns]
                st.dataframe(
                    actionable[board_cols],
                    use_container_width=True,
                    hide_index=True,
                )

                # C. Top 5 focus list
                st.markdown("**C. Top-5 focus list**")
                focus = actionable.head(5).copy()

                focus_rows = []
                for _, r in focus.iterrows():
                    score = float(r.get("Bullseye 4.0 Score", np.nan))
                    accel = float(r.get("4.0 Accelerator", np.nan))
                    beta = float(r.get("Beta 120D", np.nan))
                    core = int(r.get("4H Core Count", 0))
                    market = float(r.get("Market Regime", np.nan))

                    if score >= 95 and core >= 3:
                        confidence_note = "Highest validated confirmation tier"
                    elif score >= 95 and accel >= 10:
                        confidence_note = "Prime score with strong accelerator"
                    elif score >= 95:
                        confidence_note = "Prime 95+ score"
                    elif score >= 92.5:
                        confidence_note = "Very-high-conviction score"
                    else:
                        confidence_note = "High-conviction score"

                    regime_note = "Supportive" if pd.notna(market) and market >= 7 else "Less supportive"

                    focus_rows.append({
                        "Rank": len(focus_rows) + 1,
                        "Ticker": r["Ticker"],
                        "Action": r["4I Action"],
                        "Tier": r["4H Signal Tier"],
                        "Score": score,
                        "Price": r.get("Price", np.nan),
                        "Core": core,
                        "Accelerator": accel,
                        "Beta": beta,
                        "Market": regime_note,
                        "Why it is here": confidence_note,
                    })

                focus_df = pd.DataFrame(focus_rows)
                st.dataframe(focus_df, use_container_width=True, hide_index=True)

                # D. Decision flags: context only, no new scoring
                st.markdown("**D. Context / caution flags**")
                flag_rows = []
                for _, r in actionable.iterrows():
                    flags = []

                    rsi = pd.to_numeric(pd.Series([r.get("RSI")]), errors="coerce").iloc[0]
                    dist20 = pd.to_numeric(pd.Series([r.get("Dist 20MA %")]), errors="coerce").iloc[0]
                    relvol = pd.to_numeric(pd.Series([r.get("Rel Vol")]), errors="coerce").iloc[0]
                    regime = pd.to_numeric(pd.Series([r.get("Market Regime")]), errors="coerce").iloc[0]

                    if pd.notna(rsi) and rsi >= 80:
                        flags.append("RSI 80+")
                    if pd.notna(dist20) and dist20 >= 20:
                        flags.append("20%+ above 20MA")
                    if pd.notna(relvol) and relvol < 0.8:
                        flags.append("Light relative volume")
                    if pd.notna(regime) and regime < 7:
                        flags.append("Weaker market regime")

                    flag_rows.append({
                        "Ticker": r["Ticker"],
                        "Action": r["4I Action"],
                        "Score": r["Bullseye 4.0 Score"],
                        "Context Flags": " | ".join(flags) if flags else "None",
                    })

                flags_df = pd.DataFrame(flag_rows)
                st.dataframe(flags_df, use_container_width=True, hide_index=True)

                # E. Quick tier summary
                st.markdown("**E. Signal-tier summary**")
                tier_summary = (
                    actionable.groupby(["4I Action", "4H Signal Tier"], observed=True)
                    .agg(
                        Signals=("Ticker", "count"),
                        Avg_Score=("Bullseye 4.0 Score", "mean"),
                        Avg_Accelerator=("4.0 Accelerator", "mean"),
                        Avg_Beta=("Beta 120D", "mean"),
                        Avg_Core=("4H Core Count", "mean"),
                    )
                    .reset_index()
                )
                for c in ["Avg_Score", "Avg_Accelerator", "Avg_Beta", "Avg_Core"]:
                    tier_summary[c] = tier_summary[c].round(2)
                st.dataframe(tier_summary, use_container_width=True, hide_index=True)

                # Downloadable daily command-center watchlist.
                export_cols = [
                    "Ticker", "4I Action", "4I Action Rank", "4H Signal Tier",
                    "Bullseye 4.0 Score", "Price", "4H Core Count",
                    "4.0 Accelerator", "4H Signal Badges", "Beta 120D",
                    "Avg $ Volume 60D ($M)", "120D %", "RSI", "Dist 20MA %",
                    "Rel Vol", "RS vs SPY 20D", "Market Regime", "4I Why"
                ]
                export_cols = [c for c in export_cols if c in actionable.columns]
                now_m = pd.Timestamp.now()

                st.download_button(
                    "Download today's Phase 4M command-center watchlist",
                    actionable[export_cols].to_csv(index=False),
                    f"bullseye_phase4m_watchlist_{now_m.strftime('%Y%m%d')}.csv",
                    "text/csv",
                )
            else:
                st.warning(
                    "No 90+ high-conviction Bullseye signals are active in the broad universe right now."
                )


if run_phase4n:
    with st.spinner("Building Phase 4N entry/exit planning layer..."):
        plan_tickers = sorted(set(BROAD_TICKERS))
        data_n = download_prices(sorted(set(plan_tickers + ["SPY"])))
        spy_n = one_symbol(data_n, "SPY")
        plan_rows = []

        if spy_n is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in plan_tickers:
                df = one_symbol(data_n, t)
                if df is None:
                    continue
                try:
                    scored = score_stock(df, spy_n)
                    if scored.get("4I Action Rank", 0) < 1:
                        continue

                    plan = build_trade_plan(df, scored)
                    if plan is None:
                        continue

                    row = {
                        "Ticker": t,
                        "4I Action": scored.get("4I Action"),
                        "4I Action Rank": scored.get("4I Action Rank"),
                        "4H Signal Tier": scored.get("4H Signal Tier"),
                        "Bullseye 4.0 Score": scored.get("Bullseye 4.0 Score"),
                        "4H Core Count": scored.get("4H Core Count"),
                        "4.0 Accelerator": scored.get("4.0 Accelerator"),
                        "RSI": scored.get("RSI"),
                        "Dist 20MA %": scored.get("Dist 20MA %"),
                        "Rel Vol": scored.get("Rel Vol"),
                        "Market Regime": scored.get("Market Regime"),
                        "4H Signal Badges": scored.get("4H Signal Badges"),
                    }
                    row.update(plan)
                    plan_rows.append(row)
                except Exception:
                    continue

        if plan_rows:
            plans = pd.DataFrame(plan_rows).sort_values(
                ["4I Action Rank", "Bullseye 4.0 Score", "4H Core Count", "4.0 Accelerator"],
                ascending=[False, False, False, False],
            )

            st.subheader("🧭 Phase 4N.1 Entry / Exit Planning Layer")
            st.caption(
                "These are reference levels derived from current price structure and ATR. "
                "They do not change the Bullseye score or guarantee an entry, stop, or target."
            )

            st.markdown("**A. Live planning board**")
            display_cols = [
                "Ticker", "4I Action", "4H Signal Tier", "Bullseye 4.0 Score",
                "Entry Mode", "Current Price", "Pullback Entry Low", "Pullback Entry High",
                "Breakout Reference", "Invalidation Reference",
                "Risk / Share", "Risk %", "Risk Label", "Target 1R", "Target 2R", "Target 3R",
                "ATR14", "RSI", "Dist 20MA %", "Rel Vol", "Market Regime"
            ]
            st.dataframe(
                plans[display_cols],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**B. Top setup details**")
            for _, r in plans.head(5).iterrows():
                with st.expander(
                    f"{r['Ticker']} — {r['4I Action']} — score {r['Bullseye 4.0 Score']}"
                ):
                    detail = pd.DataFrame([
                        {
                            "Current": r["Current Price"],
                            "Entry Low": r["Pullback Entry Low"],
                            "Entry High": r["Pullback Entry High"],
                            "Breakout Ref": r["Breakout Reference"],
                            "Invalidation Ref": r["Invalidation Reference"],
                            "Target 1R": r["Target 1R"],
                            "Target 2R": r["Target 2R"],
                            "Target 3R": r["Target 3R"],
                            "ATR14": r["ATR14"],
                            "Risk %": r["Risk %"],
                            "Risk Label": r["Risk Label"],
                        }
                    ])
                    st.dataframe(detail, use_container_width=True, hide_index=True)
                    st.write(
                        f"**Mode:** {r['Entry Mode']}  \n"
                        f"**Signal:** {r['4H Signal Tier']}  \n"
                        f"**Badges:** {r['4H Signal Badges']}  \n"
                        f"**Context:** RSI {r['RSI']}, distance from 20MA {r['Dist 20MA %']}%, "
                        f"relative volume {r['Rel Vol']}, market regime {r['Market Regime']}."
                    )

            st.markdown("**C. Entry-mode summary**")
            mode_summary = (
                plans.groupby("Entry Mode", observed=True)
                .agg(
                    Signals=("Ticker", "count"),
                    Avg_Score=("Bullseye 4.0 Score", "mean"),
                    Avg_Risk_Pct=("Risk %", "mean"),
                    Avg_ATR=("ATR14", "mean"),
                )
                .reset_index()
            )
            for c in ["Avg_Score", "Avg_Risk_Pct", "Avg_ATR"]:
                mode_summary[c] = mode_summary[c].round(2)
            st.dataframe(mode_summary, use_container_width=True, hide_index=True)

            export_cols = [
                "Ticker", "4I Action", "4H Signal Tier", "Bullseye 4.0 Score",
                "Entry Mode", "Current Price", "Pullback Entry Low", "Pullback Entry High",
                "Breakout Reference", "Invalidation Reference", "Risk / Share", "Risk %",
                "Target 1R", "Target 2R", "Target 3R", "ATR14",
                "4H Core Count", "4.0 Accelerator", "RSI", "Dist 20MA %",
                "Rel Vol", "Market Regime", "4H Signal Badges"
            ]
            now_n = pd.Timestamp.now()
            st.download_button(
                "Download today's Phase 4N trade-planning sheet",
                plans[export_cols].to_csv(index=False),
                f"bullseye_phase4n_trade_plan_{now_n.strftime('%Y%m%d')}.csv",
                "text/csv",
            )
        else:
            st.warning("No active 90+ Bullseye signals are available for planning right now.")

st.caption(f"Phase 4N.1 generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.")

# Phase 4Q.9D — durable Closed Trades History
if _phase4q5_storage_config()["configured"]:
    try:
        closed_main_rows = _phase4q9_load_closed_trades(100)
    except Exception:
        closed_main_rows = []
    if closed_main_rows:
        st.subheader("🗂️ Phase 4Q.9F Closed Trades History")
        st.caption("Completed Bullseye trades preserved for forward validation and future performance analysis.")
        table_rows=[]
        for r in closed_main_rows:
            e=float(r.get("entry") or 0); x=float(r.get("final_exit_price") or 0)
            s=float(r.get("initial_shares") or 0); p=float(r.get("realized_pl") or 0)
            price_return_pct=((x/e)-1)*100 if e>0 and x>0 else None
            hr=r.get("highest_r")
            table_rows.append({
                "Ticker":str(r.get("ticker") or "").upper(),
                "Entry":f"${e:,.2f}","Exit":f"${x:,.2f}",
                "Initial Shares":f"{s:.5f}","Realized P/L":f"${p:,.2f}",
                "Price Return":f"{price_return_pct:+.2f}%" if price_return_pct is not None else "—",
                "Highest R":f"{float(hr):.2f}R" if hr is not None else "—",
                "Highest State":str(r.get("highest_state") or "—"),
                "Exit Reason":str(r.get("exit_reason") or "—"),
                "Closed":str(r.get("closed_at") or "")[:10] or "—",
            })
        st.dataframe(pd.DataFrame(table_rows),use_container_width=True,hide_index=True)
        vals=[float(r.get("realized_pl") or 0) for r in closed_main_rows]
        winners=sum(v>0 for v in vals); total=len(vals)
        a,b,c,d=st.columns(4)
        a.metric("Closed Trades",total); b.metric("Winners",winners)
        c.metric("Win Rate",f"{(winners/total*100 if total else 0):.1f}%")
        d.metric("Total Realized P/L",f"${sum(vals):,.2f}")


if run_phase4o:
    with st.spinner("Building Phase 4O position-sizing and trade-construction layer..."):
        sizing_tickers = sorted(set(BROAD_TICKERS))
        data_o = download_prices(sorted(set(sizing_tickers + ["SPY"])))
        spy_o = one_symbol(data_o, "SPY")
        sizing_rows = []

        if spy_o is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in sizing_tickers:
                df = one_symbol(data_o, t)
                if df is None:
                    continue
                try:
                    scored = score_stock(df, spy_o)
                    if scored.get("4I Action Rank", 0) < 1:
                        continue

                    plan = build_trade_plan(df, scored)
                    if plan is None:
                        continue

                    entry_price = (
                        float(plan["Pullback Entry Low"]) + float(plan["Pullback Entry High"])
                    ) / 2.0
                    stop_price = float(plan["Invalidation Reference"])
                    risk_per_share = max(entry_price - stop_price, 0.01)

                    risk_budget = float(phase4o_account_size) * float(phase4o_risk_pct) / 100.0
                    shares_by_risk = int(risk_budget // risk_per_share)

                    max_position_dollars = (
                        float(phase4o_account_size) * float(phase4o_max_position_pct) / 100.0
                    )
                    shares_by_concentration = int(max_position_dollars // entry_price) if entry_price > 0 else 0

                    suggested_shares = max(
                        0, min(shares_by_risk, shares_by_concentration)
                    )
                    position_value = suggested_shares * entry_price
                    actual_dollar_risk = suggested_shares * risk_per_share
                    actual_account_risk_pct = (
                        actual_dollar_risk / float(phase4o_account_size) * 100.0
                        if phase4o_account_size else 0.0
                    )
                    account_deployed_pct = (
                        position_value / float(phase4o_account_size) * 100.0
                        if phase4o_account_size else 0.0
                    )

                    concentration_limited = (
                        shares_by_concentration < shares_by_risk and shares_by_risk > 0
                    )

                    if suggested_shares < 1:
                        sizing_status = "No fit"
                    elif concentration_limited:
                        sizing_status = "Concentration capped"
                    elif plan["Risk Label"] == "Wide":
                        sizing_status = "Wide-risk setup"
                    else:
                        sizing_status = "Fits risk budget"

                    row = {
                        "Ticker": t,
                        "4I Action": scored.get("4I Action"),
                        "4H Signal Tier": scored.get("4H Signal Tier"),
                        "Bullseye 4.0 Score": scored.get("Bullseye 4.0 Score"),
                        "Entry Mode": plan.get("Entry Mode"),
                        "Planning Entry": round(entry_price, 2),
                        "Invalidation Reference": round(stop_price, 2),
                        "Risk / Share": round(risk_per_share, 2),
                        "4N Risk %": plan.get("Risk %"),
                        "4N Risk Label": plan.get("Risk Label"),
                        "Risk Budget $": round(risk_budget, 2),
                        "Shares by Risk": shares_by_risk,
                        "Shares by Concentration": shares_by_concentration,
                        "Suggested Shares": suggested_shares,
                        "Position Value $": round(position_value, 2),
                        "Account Deployed %": round(account_deployed_pct, 2),
                        "Actual $ Risk": round(actual_dollar_risk, 2),
                        "Actual Account Risk %": round(actual_account_risk_pct, 2),
                        "Sizing Status": sizing_status,
                        "Target 1R": plan.get("Target 1R"),
                        "Target 2R": plan.get("Target 2R"),
                        "Target 3R": plan.get("Target 3R"),
                    }
                    sizing_rows.append(row)
                except Exception:
                    continue

        if sizing_rows:
            sizes = pd.DataFrame(sizing_rows).sort_values(
                ["4I Action", "Bullseye 4.0 Score"],
                ascending=[True, False],
            )

            st.subheader("🧮 Phase 4O Position Sizing / Trade Construction")
            st.caption(
                "Sizing is based on the Phase 4N.1 planning-entry midpoint and technical invalidation. "
                "The technical stop is never widened or tightened to force a preferred share count."
            )

            st.markdown("**A. Portfolio risk settings**")
            settings_df = pd.DataFrame([{
                "Account Size $": round(float(phase4o_account_size), 2),
                "Max Risk / Trade %": round(float(phase4o_risk_pct), 2),
                "Dollar Risk Budget $": round(
                    float(phase4o_account_size) * float(phase4o_risk_pct) / 100.0, 2
                ),
                "Max Position %": int(phase4o_max_position_pct),
                "Max Position $": round(
                    float(phase4o_account_size) * float(phase4o_max_position_pct) / 100.0, 2
                ),
            }])
            st.dataframe(settings_df, use_container_width=True, hide_index=True)

            st.markdown("**B. Live position-sizing board**")
            sizing_cols = [
                "Ticker", "4I Action", "4H Signal Tier", "Bullseye 4.0 Score",
                "Planning Entry", "Invalidation Reference", "Risk / Share",
                "4N Risk %", "4N Risk Label", "Risk Budget $",
                "Suggested Shares", "Position Value $", "Account Deployed %",
                "Actual $ Risk", "Actual Account Risk %", "Sizing Status",
            ]
            st.dataframe(
                sizes[sizing_cols],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**C. Trade-construction details**")
            for _, r in sizes.head(5).iterrows():
                with st.expander(
                    f"{r['Ticker']} — {r['4I Action']} — {r['Suggested Shares']} shares"
                ):
                    details = pd.DataFrame([{
                        "Planning Entry": r["Planning Entry"],
                        "Invalidation": r["Invalidation Reference"],
                        "Risk / Share": r["Risk / Share"],
                        "Shares by Risk": r["Shares by Risk"],
                        "Shares by Concentration": r["Shares by Concentration"],
                        "Suggested Shares": r["Suggested Shares"],
                        "Position Value $": r["Position Value $"],
                        "Actual $ Risk": r["Actual $ Risk"],
                        "Actual Account Risk %": r["Actual Account Risk %"],
                        "Target 1R": r["Target 1R"],
                        "Target 2R": r["Target 2R"],
                        "Target 3R": r["Target 3R"],
                        "Status": r["Sizing Status"],
                    }])
                    st.dataframe(details, use_container_width=True, hide_index=True)

            st.markdown("**D. Portfolio concentration check**")
            total_position_value = float(sizes["Position Value $"].sum())
            total_open_risk = float(sizes["Actual $ Risk"].sum())
            portfolio_summary = pd.DataFrame([{
                "Actionable Setups": int(len(sizes)),
                "Combined Position Value $": round(total_position_value, 2),
                "Combined Capital %": round(
                    total_position_value / float(phase4o_account_size) * 100.0, 2
                ) if phase4o_account_size else 0.0,
                "Combined Open Risk $": round(total_open_risk, 2),
                "Combined Open Risk %": round(
                    total_open_risk / float(phase4o_account_size) * 100.0, 2
                ) if phase4o_account_size else 0.0,
            }])
            st.dataframe(portfolio_summary, use_container_width=True, hide_index=True)

            if total_position_value > float(phase4o_account_size):
                st.warning(
                    "The combined suggested positions exceed the account size. "
                    "Treat the rows as individual trade plans, not a recommendation to open every setup simultaneously."
                )
            elif (
                total_position_value / float(phase4o_account_size) * 100.0
                > 75.0
            ):
                st.warning(
                    "Opening every listed setup simultaneously would deploy more than 75% of the account."
                )

            export_cols_o = [
                "Ticker", "4I Action", "4H Signal Tier", "Bullseye 4.0 Score",
                "Entry Mode", "Planning Entry", "Invalidation Reference",
                "Risk / Share", "4N Risk %", "4N Risk Label", "Risk Budget $",
                "Shares by Risk", "Shares by Concentration", "Suggested Shares",
                "Position Value $", "Account Deployed %", "Actual $ Risk",
                "Actual Account Risk %", "Sizing Status",
                "Target 1R", "Target 2R", "Target 3R",
            ]
            now_o = pd.Timestamp.now()
            st.download_button(
                "Download today's Phase 4O position-sizing sheet",
                sizes[export_cols_o].to_csv(index=False),
                f"bullseye_phase4o_position_sizing_{now_o.strftime('%Y%m%d')}.csv",
                "text/csv",
            )
        else:
            st.warning("No active Bullseye signals are available for position sizing right now.")

st.caption(f"Phase 4O generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.")


if run_phase4p:
    with st.spinner("Building Phase 4P portfolio-risk plan..."):
        portfolio_tickers = sorted(set(BROAD_TICKERS))
        data_p = download_prices(sorted(set(portfolio_tickers + ["SPY"])))
        spy_p = one_symbol(data_p, "SPY")
        portfolio_rows = []

        if spy_p is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in portfolio_tickers:
                df = one_symbol(data_p, t)
                if df is None:
                    continue
                try:
                    scored = score_stock(df, spy_p)
                    if scored.get("4I Action Rank", 0) < 1:
                        continue

                    plan = build_trade_plan(df, scored)
                    if plan is None:
                        continue

                    entry_price = (
                        float(plan["Pullback Entry Low"]) + float(plan["Pullback Entry High"])
                    ) / 2.0
                    stop_price = float(plan["Invalidation Reference"])
                    risk_per_share = max(entry_price - stop_price, 0.01)

                    risk_budget = float(phase4o_account_size) * float(phase4o_risk_pct) / 100.0
                    shares_by_risk = int(risk_budget // risk_per_share)

                    max_position_dollars = (
                        float(phase4o_account_size) * float(phase4o_max_position_pct) / 100.0
                    )
                    shares_by_concentration = int(max_position_dollars // entry_price) if entry_price > 0 else 0
                    base_shares = max(0, min(shares_by_risk, shares_by_concentration))

                    portfolio_rows.append({
                        "Ticker": t,
                        "4I Action": scored.get("4I Action"),
                        "4I Action Rank": scored.get("4I Action Rank", 0),
                        "4H Signal Tier": scored.get("4H Signal Tier"),
                        "Bullseye 4.0 Score": scored.get("Bullseye 4.0 Score"),
                        "Planning Entry": round(entry_price, 2),
                        "Invalidation Reference": round(stop_price, 2),
                        "Risk / Share": round(risk_per_share, 2),
                        "4N Risk %": plan.get("Risk %"),
                        "4N Risk Label": plan.get("Risk Label"),
                        "4O Base Shares": base_shares,
                        "4O Base Position $": round(base_shares * entry_price, 2),
                        "4O Base Risk $": round(base_shares * risk_per_share, 2),
                    })
                except Exception:
                    continue

        if portfolio_rows:
            pf = pd.DataFrame(portfolio_rows).sort_values(
                ["4I Action Rank", "Bullseye 4.0 Score"],
                ascending=[False, False],
            ).reset_index(drop=True)

            cluster_map, corr = build_corr_clusters(
                data_p,
                pf["Ticker"].tolist(),
                corr_threshold=float(phase4p_corr_threshold),
                lookback=60,
            )
            pf["Correlation Cluster"] = pf["Ticker"].map(cluster_map).fillna("Solo")

            total_risk_cap = float(phase4o_account_size) * float(phase4p_max_total_risk_pct) / 100.0
            cluster_risk_cap = float(phase4o_account_size) * float(phase4p_max_cluster_risk_pct) / 100.0

            remaining_total_risk = total_risk_cap
            cluster_used = {}
            adjusted_rows = []

            for _, r in pf.iterrows():
                cluster = r["Correlation Cluster"]
                cluster_used.setdefault(cluster, 0.0)

                risk_per_share = float(r["Risk / Share"])
                base_shares = int(r["4O Base Shares"])

                cluster_remaining = max(cluster_risk_cap - cluster_used[cluster], 0.0)
                total_remaining = max(remaining_total_risk, 0.0)

                shares_by_cluster = int(cluster_remaining // risk_per_share) if risk_per_share > 0 else 0
                shares_by_total = int(total_remaining // risk_per_share) if risk_per_share > 0 else 0
                adjusted_shares = max(0, min(base_shares, shares_by_cluster, shares_by_total))

                adjusted_risk = adjusted_shares * risk_per_share
                adjusted_position = adjusted_shares * float(r["Planning Entry"])

                cluster_used[cluster] += adjusted_risk
                remaining_total_risk -= adjusted_risk

                if adjusted_shares == 0 and base_shares > 0:
                    portfolio_status = "Blocked by portfolio cap"
                elif adjusted_shares < base_shares:
                    portfolio_status = "Reduced by portfolio cap"
                else:
                    portfolio_status = "Unchanged"

                adjusted_rows.append({
                    **r.to_dict(),
                    "4P Shares": adjusted_shares,
                    "4P Position $": round(adjusted_position, 2),
                    "4P Risk $": round(adjusted_risk, 2),
                    "4P Account Risk %": round(
                        adjusted_risk / float(phase4o_account_size) * 100.0, 2
                    ) if phase4o_account_size else 0.0,
                    "4P Status": portfolio_status,
                })

            p4p = pd.DataFrame(adjusted_rows)

            st.subheader("🧩 Phase 4P Portfolio-Risk Layer")
            st.caption(
                "4P does not change Bullseye ranking, entry zones, or technical invalidation. "
                "It only reduces share counts when total or correlated-cluster risk would become excessive."
            )

            st.markdown("**A. Portfolio guardrails**")
            guardrails = pd.DataFrame([{
                "Account Size $": round(float(phase4o_account_size), 2),
                "Individual Risk / Trade %": round(float(phase4o_risk_pct), 2),
                "Max Combined Open Risk %": round(float(phase4p_max_total_risk_pct), 2),
                "Max Combined Open Risk $": round(total_risk_cap, 2),
                "Correlation Threshold": round(float(phase4p_corr_threshold), 2),
                "Max Risk / Correlation Cluster %": round(float(phase4p_max_cluster_risk_pct), 2),
                "Max Risk / Correlation Cluster $": round(cluster_risk_cap, 2),
            }])
            st.dataframe(guardrails, use_container_width=True, hide_index=True)

            st.markdown("**B. 4O vs 4P share sizing**")
            compare_cols = [
                "Ticker", "4I Action", "4H Signal Tier", "Bullseye 4.0 Score",
                "Correlation Cluster", "Planning Entry", "Invalidation Reference",
                "Risk / Share", "4O Base Shares", "4O Base Risk $",
                "4P Shares", "4P Risk $", "4P Account Risk %", "4P Status"
            ]
            st.dataframe(
                p4p[compare_cols],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**C. Correlation-cluster exposure**")
            cluster_summary = (
                p4p.groupby("Correlation Cluster", observed=True)
                .agg(
                    Tickers=("Ticker", lambda x: ", ".join(x)),
                    Positions=("Ticker", "count"),
                    Portfolio_Risk=("4P Risk $", "sum"),
                    Position_Value=("4P Position $", "sum"),
                )
                .reset_index()
            )
            cluster_summary["Cluster Risk %"] = (
                cluster_summary["Portfolio_Risk"] / float(phase4o_account_size) * 100.0
            ).round(2)
            cluster_summary["Capital %"] = (
                cluster_summary["Position_Value"] / float(phase4o_account_size) * 100.0
            ).round(2)
            cluster_summary["Portfolio_Risk"] = cluster_summary["Portfolio_Risk"].round(2)
            cluster_summary["Position_Value"] = cluster_summary["Position_Value"].round(2)
            st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

            st.markdown("**D. Pairwise correlation matrix (60 trading days)**")
            if not corr.empty:
                st.dataframe(corr.round(2), use_container_width=True)
            else:
                st.info("Not enough overlapping price history to calculate correlations.")

            st.markdown("**E. Portfolio-level risk summary**")
            total_4o_risk = float(p4p["4O Base Risk $"].sum())
            total_4p_risk = float(p4p["4P Risk $"].sum())
            total_4p_position = float(p4p["4P Position $"].sum())
            reduced_count = int((p4p["4P Shares"] < p4p["4O Base Shares"]).sum())

            summary = pd.DataFrame([{
                "Actionable Setups": len(p4p),
                "4O Combined Risk $": round(total_4o_risk, 2),
                "4O Combined Risk %": round(
                    total_4o_risk / float(phase4o_account_size) * 100.0, 2
                ) if phase4o_account_size else 0.0,
                "4P Combined Risk $": round(total_4p_risk, 2),
                "4P Combined Risk %": round(
                    total_4p_risk / float(phase4o_account_size) * 100.0, 2
                ) if phase4o_account_size else 0.0,
                "4P Capital Deployed $": round(total_4p_position, 2),
                "4P Capital Deployed %": round(
                    total_4p_position / float(phase4o_account_size) * 100.0, 2
                ) if phase4o_account_size else 0.0,
                "Positions Reduced / Blocked": reduced_count,
            }])
            st.dataframe(summary, use_container_width=True, hide_index=True)

            if total_4p_risk > total_risk_cap + 0.01:
                st.error("Portfolio risk exceeds the configured 4P total-risk cap.")
            elif reduced_count:
                st.warning(
                    "4P reduced one or more positions to respect the configured portfolio/correlation risk caps."
                )
            else:
                st.success("The current 4O position sizes already fit inside the 4P portfolio-risk guardrails.")

            now_p = pd.Timestamp.now()
            export_cols_p = [
                "Ticker", "4I Action", "4H Signal Tier", "Bullseye 4.0 Score",
                "Correlation Cluster", "Planning Entry", "Invalidation Reference",
                "Risk / Share", "4N Risk %", "4N Risk Label",
                "4O Base Shares", "4O Base Position $", "4O Base Risk $",
                "4P Shares", "4P Position $", "4P Risk $",
                "4P Account Risk %", "4P Status",
            ]
            st.download_button(
                "Download today's Phase 4P portfolio-risk plan",
                p4p[export_cols_p].to_csv(index=False),
                f"bullseye_phase4p_portfolio_risk_{now_p.strftime('%Y%m%d')}.csv",
                "text/csv",
            )
        else:
            st.warning("No active Bullseye signals are available for Phase 4P portfolio-risk planning.")

st.caption(f"Phase 4P generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.")

if run_phase4q:
    with st.spinner("Building Phase 4Q active trade-management plans..."):
        q_tickers = sorted(set(BROAD_TICKERS))
        data_q = download_prices(sorted(set(q_tickers + ["SPY"])))
        spy_q = one_symbol(data_q, "SPY")
        q_rows = []

        if spy_q is None:
            st.error("Could not retrieve SPY data.")
        else:
            for t in q_tickers:
                df = one_symbol(data_q, t)
                if df is None:
                    continue
                try:
                    scored = score_stock(df, spy_q)
                    if scored.get("4I Action Rank", 0) < 1:
                        continue

                    trade_plan = build_trade_plan(df, scored)
                    if trade_plan is None:
                        continue

                    mgmt = build_trade_management_plan(
                        df,
                        trade_plan,
                        trim_pct=phase4q_trim_pct,
                        trail_start_r=phase4q_trail_start_r,
                        trail_atr=phase4q_trail_atr,
                    )
                    if mgmt is None:
                        continue

                    row = {
                        "Ticker": t,
                        "4I Action": scored.get("4I Action"),
                        "4I Action Rank": scored.get("4I Action Rank"),
                        "4H Signal Tier": scored.get("4H Signal Tier"),
                        "Bullseye 4.0 Score": scored.get("Bullseye 4.0 Score"),
                        "4H Core Count": scored.get("4H Core Count"),
                        "4.0 Accelerator": scored.get("4.0 Accelerator"),
                        "4H Signal Badges": scored.get("4H Signal Badges"),
                        "RSI": scored.get("RSI"),
                        "Dist 20MA %": scored.get("Dist 20MA %"),
                        "Rel Vol": scored.get("Rel Vol"),
                        "Market Regime": scored.get("Market Regime"),
                        "Entry Mode": trade_plan.get("Entry Mode"),
                        "Risk %": trade_plan.get("Risk %"),
                        "Risk Label": trade_plan.get("Risk Label"),
                        "ATR14": trade_plan.get("ATR14"),
                    }
                    row.update(mgmt)
                    q_rows.append(row)
                except Exception:
                    continue

        if q_rows:
            q = pd.DataFrame(q_rows).sort_values(
                ["4I Action Rank", "Bullseye 4.0 Score", "Current R"],
                ascending=[False, False, False],
            )

            st.subheader("🧠 Phase 4Q Active Trade-Management Planner")
            st.caption(
                "Reference guidance only. Phase 4Q does not alter Bullseye scoring, entry logic, sizing, or portfolio-risk limits."
            )

            st.markdown("**A. Active management board**")
            cols = [
                "Ticker", "4I Action", "Bullseye 4.0 Score", "Entry Mode",
                "Entry Reference", "Current Price", "Current R",
                "Initial Stop", "Target 1", "Target 2", "Target 3",
                "R:R to T1", "R:R to T2", "R:R to T3",
                "Management Action", "Partial Profit %",
                "Breakeven Trigger", "Trail Trigger", "ATR Trail Ref",
                "Risk %", "Risk Label",
            ]
            st.dataframe(q[[c for c in cols if c in q.columns]], use_container_width=True, hide_index=True)

            st.markdown("**B. Management-action summary**")
            summary = (
                q.groupby("Management Action", observed=True)
                .agg(
                    Positions=("Ticker", "count"),
                    Avg_Score=("Bullseye 4.0 Score", "mean"),
                    Avg_Current_R=("Current R", "mean"),
                    Avg_Risk_Pct=("Risk %", "mean"),
                )
                .reset_index()
            )
            for c in ["Avg_Score", "Avg_Current_R", "Avg_Risk_Pct"]:
                summary[c] = summary[c].round(2)
            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.markdown("**C. Top setup management details**")
            for _, r in q.head(5).iterrows():
                with st.expander(f"{r['Ticker']} — {r['4I Action']} — {r['Management Action']}"):
                    detail = pd.DataFrame([{
                        "Entry": r["Entry Reference"],
                        "Current": r["Current Price"],
                        "Current R": r["Current R"],
                        "Initial Stop": r["Initial Stop"],
                        "T1": r["Target 1"],
                        "T2": r["Target 2"],
                        "T3": r["Target 3"],
                        "Breakeven Trigger": r["Breakeven Trigger"],
                        "Trail Trigger": r["Trail Trigger"],
                        "ATR Trail Ref": r["ATR Trail Ref"],
                    }])
                    st.dataframe(detail, use_container_width=True, hide_index=True)
                    st.write(
                        f"**Management:** {r['Management Action']}  \n"
                        f"**Reason:** {r['Management Reason']}  \n"
                        f"**Signal:** {r['4H Signal Tier']}  \n"
                        f"**Badges:** {r['4H Signal Badges']}"
                    )

            st.markdown("**D. Management framework**")
            framework = pd.DataFrame([
                {"Condition": "Below entry, above invalidation", "Default posture": "Hold / monitor", "Purpose": "Allow normal trade noise."},
                {"Condition": "At +1R", "Default posture": "Protect", "Purpose": "Consider moving risk toward breakeven."},
                {"Condition": "At Target 1 / +1R", "Default posture": f"Trim {phase4q_trim_pct}%", "Purpose": "Bank part of the gain."},
                {"Condition": f"At +{phase4q_trail_start_r:.1f}R or better", "Default posture": "Trail", "Purpose": f"Use about {phase4q_trail_atr:.2f} ATR to protect a winner."},
                {"Condition": "At +2R or better", "Default posture": "Trim / trail remainder", "Purpose": "Protect a mature swing while preserving upside."},
                {"Condition": "At/below invalidation", "Default posture": "Exit", "Purpose": "Respect the original thesis failure point."},
            ])
            st.dataframe(framework, use_container_width=True, hide_index=True)

            now_q = pd.Timestamp.now()
            st.download_button(
                "Download Phase 4Q trade-management plans",
                q.to_csv(index=False),
                f"bullseye_phase4q_trade_management_{now_q.strftime('%Y%m%d')}.csv",
                "text/csv",
            )
        else:
            st.warning("No actionable Phase 4Q trade-management plans were returned.")



if run_phase4q1 or st.session_state.get("phase4q1_view_active", False):
    if (
        st.session_state.get("phase4q8_promotion_active", False)
        and st.session_state.get("phase4q1_state_key") == "Entered / Live Position"
        and str(st.session_state.get("phase4q1_ticker_key", "")).upper().strip()
        == str(st.session_state.get("phase4q8_promotion_ticker", "")).upper().strip()
    ):
        snap = st.session_state.get("phase4q8_promotion_snapshot", {}) or {}
        pt = str(st.session_state.get("phase4q8_promotion_ticker", "")).upper().strip()

        st.markdown(f"### 🔎 Promotion Reference — {pt}")
        st.info(
            "Saved Candidate / Watching snapshot for reference while entering the actual trade. "
            "These are investigative levels, not execution data. This panel clears after a successful Save / Update Position."
        )

        pr1, pr2, pr3, pr4 = st.columns(4)
        pr1.metric("Candidate Action", str(snap.get("candidate_action") or "—"))
        score = snap.get("bullseye_score")
        pr2.metric("Bullseye Score", f"{float(score):.1f}" if score is not None else "—")
        pr3.metric("Signal Tier", str(snap.get("signal_tier") or "—"))
        mark = snap.get("current_mark")
        pr4.metric("Candidate Mark", f"${float(mark):,.2f}" if mark is not None else "—")

        why = str(snap.get("action_reason") or "").strip()
        if why:
            st.markdown(f"**Why:** {why}")

        refs = [
            ("Preferred Entry Low", snap.get("entry_low"), "Lower edge of saved candidate entry zone"),
            ("Preferred Entry High", snap.get("entry_high"), "Upper edge of saved candidate entry zone"),
            ("Breakout Reference", snap.get("breakout_reference"), "Continuation / breakout reference"),
            ("Invalidation Reference", snap.get("invalidation_reference"), "Technical thesis-failure reference"),
            ("+1R Target", snap.get("target_1r"), "First profit-protection reference"),
            ("+2R Target", snap.get("target_2r"), "Partial-profit / trailing reference"),
            ("+3R Target", snap.get("target_3r"), "Winner-protection reference"),
        ]
        st.dataframe(
            pd.DataFrame([
                {
                    "Level": label,
                    "Price": f"${float(value):,.2f}" if value is not None else "—",
                    "Meaning": meaning,
                }
                for label, value, meaning in refs
            ]),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Enter the broker-confirmed fill, shares, realized P/L (normally $0 at entry), "
            "and original stop in the live-position inputs."
        )
        st.markdown("---")

    st.subheader("📍 Phase 4Q.1 Position-State Manager")
    st.caption(
        "This layer does not change Bullseye 4.0 scoring or Phase 4Q management math. "
        "Candidate / Watching investigates pre-entry setups; Entered / Live Position manages the actual trade you entered."
    )

    state = phase4q1_state
    ticker = phase4q1_ticker

    if state == "Candidate / Watching":
        st.info(
            "Candidate / Watching is Bullseye's pre-entry investigative mode. "
            "No ownership or actual P/L is assumed; all entry, invalidation and target levels "
            "remain Bullseye reference levels."
        )

        if not ticker:
            st.warning("Enter a ticker to investigate, then run the Position-State Manager.")
        else:
            with st.spinner(f"Investigating {ticker} with current Bullseye references..."):
                data_cand = download_prices(sorted(set([ticker, "SPY"])))
                df_cand = one_symbol(data_cand, ticker)
                spy_cand = one_symbol(data_cand, "SPY")

            if df_cand is None or spy_cand is None:
                st.error("Could not retrieve enough market data for this candidate.")
            else:
                candidate_stage = "starting candidate investigation"
                try:
                    candidate_stage = "score_stock"
                    scored_cand = score_stock(df_cand, spy_cand)
                    candidate_stage = "4R early-warning classification"
                    scored_cand.update(build_phase4r_early_warning(scored_cand))
                    candidate_stage = "build_trade_plan"
                    plan_cand = build_trade_plan(df_cand, scored_cand)

                    if plan_cand is None:
                        st.error("Bullseye could not build a technical trade plan for this candidate.")
                    else:
                        daily_cand = float(df_cand["Close"].iloc[-1])
                        candidate_stage = "get_position_mark"
                        mark_cand_info = get_position_mark(ticker)

                        if (
                            mark_cand_info.get("status") == "OK"
                            and pd.notna(mark_cand_info.get("price"))
                        ):
                            mark_cand = float(mark_cand_info["price"])
                            cand_session = mark_cand_info.get("session", "Current")
                            cand_timestamp = mark_cand_info.get("timestamp")
                            cand_source = mark_cand_info.get("source", "Yahoo/yfinance")
                        else:
                            mark_cand = daily_cand
                            cand_session = "Daily close fallback"
                            cand_timestamp = df_cand.index[-1]
                            cand_source = "Bullseye daily close fallback"

                        candidate_stage = "build_candidate_investigation"
                        investigation = build_candidate_investigation(
                            scored_cand,
                            plan_cand,
                            mark_cand,
                        )
                        opportunity_4r3 = build_phase4r3_opportunity_state(
                            scored_cand,
                            plan_cand,
                            mark_cand,
                        )
                        market_cap_cand = get_market_cap(ticker)
                        market_cap_cand_display = _format_market_cap(market_cap_cand)

                        candidate_stage = "render candidate investigation"
                        st.subheader(f"🔎 Candidate Investigation — {ticker}")

                        r_stage = str(scored_cand.get("4R Stage", "Unavailable"))
                        r_gap = float(scored_cand.get("4R Gap to 90", np.nan))
                        r_ready = int(scored_cand.get("4R Readiness", 0) or 0)
                        rw1, rw2, rw3 = st.columns(3)
                        rw1.metric("4R Stage", r_stage)
                        rw2.metric("Gap to 90", f"{r_gap:.1f}" if np.isfinite(r_gap) else "—")
                        rw3.metric("4R Readiness", r_ready)
                        st.caption(str(scored_cand.get("4R Why", "")))
                        st.caption("Next trigger: " + str(scored_cand.get("4R Next Trigger", "")))

                        st.markdown("#### 🎯 Phase 4R.3 Opportunity State")
                        o1, o2, o3, o4 = st.columns(4)
                        o1.metric("Opportunity", opportunity_4r3["4R.3 Opportunity Signal"])
                        o2.metric("4R Stage", r_stage)
                        o3.metric("Current Mark", f"${mark_cand:,.2f}")
                        o4.metric("Market Cap", market_cap_cand_display)
                        st.markdown(f'**Why:** {opportunity_4r3["4R.3 Why"]}')
                        st.caption("Next trigger: " + str(opportunity_4r3["4R.3 Next Trigger"]))

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Candidate Action", investigation["Candidate Action"])
                        c2.metric("Bullseye 4.0 Score", f'{float(scored_cand.get("Bullseye 4.0 Score", np.nan)):.1f}')
                        c3.metric("Signal Tier", str(scored_cand.get("4H Signal Tier", "—")))
                        c4.metric("Current Mark", f"${mark_cand:,.2f}")

                        st.markdown(f'**Existing Candidate Why:** {investigation["Action Reason"]}')

                        candidate_summary = pd.DataFrame([{
                            "Ticker": ticker,
                            "Market Cap": market_cap_cand_display,
                            "Market Cap ($)": market_cap_cand,
                            "4R.3 Opportunity": opportunity_4r3.get("4R.3 Opportunity Signal"),
                            "4R.3 Why": opportunity_4r3.get("4R.3 Why"),
                            "4R.3 Next Trigger": opportunity_4r3.get("4R.3 Next Trigger"),
                            "Bullseye Action": scored_cand.get("4I Action"),
                            "Signal Tier": scored_cand.get("4H Signal Tier"),
                            "Bullseye 4.0 Score": scored_cand.get("Bullseye 4.0 Score"),
                            "Setup Quality": scored_cand.get("Setup Quality"),
                            "RSI": scored_cand.get("RSI"),
                            "Dist 20MA %": scored_cand.get("Dist 20MA %"),
                            "Rel Vol": scored_cand.get("Rel Vol"),
                            "RS vs SPY 20D": scored_cand.get("RS vs SPY 20D"),
                            "Market Regime": scored_cand.get("Market Regime"),
                            "Entry Mode": plan_cand.get("Entry Mode"),
                            "Entry Zone Position": investigation.get("Entry Zone Position"),
                            "Entry Distance %": investigation.get("Entry Distance %"),
                        }])

                        st.markdown("**A. Setup quality / timing**")
                        st.dataframe(candidate_summary, use_container_width=True, hide_index=True)

                        levels = pd.DataFrame([
                            {"Level": "Current Mark", "Price": mark_cand, "Meaning": f"{cand_session} candidate mark"},
                            {"Level": "Pullback Entry Low", "Price": plan_cand.get("Pullback Entry Low"), "Meaning": "Lower edge of preferred Bullseye entry zone"},
                            {"Level": "Pullback Entry High", "Price": plan_cand.get("Pullback Entry High"), "Meaning": "Upper edge of preferred Bullseye entry zone"},
                            {"Level": "Breakout Reference", "Price": plan_cand.get("Breakout Reference"), "Meaning": "Continuation / breakout reference"},
                            {"Level": "Invalidation Reference", "Price": plan_cand.get("Invalidation Reference"), "Meaning": "Technical thesis-failure reference"},
                            {"Level": "+1R Target", "Price": plan_cand.get("Target 1R"), "Meaning": "First profit-protection reference"},
                            {"Level": "+2R Target", "Price": plan_cand.get("Target 2R"), "Meaning": "Partial-profit / trailing reference"},
                            {"Level": "+3R Target", "Price": plan_cand.get("Target 3R"), "Meaning": "Winner-protection reference"},
                        ])

                        st.markdown("**B. Bullseye pre-entry levels**")
                        st.dataframe(
                            levels,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Price": st.column_config.NumberColumn(format="$%.2f"),
                            },
                        )

                        risk_row = pd.DataFrame([{
                            "Risk / Share": plan_cand.get("Risk / Share"),
                            "Risk %": plan_cand.get("Risk %"),
                            "Risk Label": plan_cand.get("Risk Label"),
                            "ATR14": plan_cand.get("ATR14"),
                            "Breakout Distance %": plan_cand.get("Breakout Distance %"),
                            "Position State": "Candidate / Watching",
                            "Ownership Assumed": "No",
                            "Actual P/L": "Disabled",
                        }])

                        st.markdown("**C. Risk / investigation context**")
                        st.dataframe(risk_row, use_container_width=True, hide_index=True)

                        if cand_timestamp is not None:
                            try:
                                cand_ts_text = pd.Timestamp(cand_timestamp).strftime("%Y-%m-%d %H:%M:%S %Z")
                            except Exception:
                                cand_ts_text = str(cand_timestamp)
                        else:
                            cand_ts_text = "Unavailable"

                        st.caption(
                            f"Candidate mark source: {cand_source} | Session: {cand_session} | "
                            f"Timestamp: {cand_ts_text}. "
                            "Candidate watchlist saves are separate from live-position storage."
                        )

                        if _phase4q5_storage_config()["configured"]:
                            if st.button(
                                f"👀 Save {ticker} to Candidate Watchlist",
                                key=f"phase4q8_save_{ticker}",
                            ):
                                try:
                                    phase4q8_saved = _phase4q8_save_candidate(
                                        ticker,
                                        scored_cand,
                                        plan_cand,
                                        investigation,
                                        mark_cand,
                                    )
                                    _phase4r5_start_or_refresh_candidate(ticker, scored_cand, plan_cand, mark_cand)
                                    if phase4q8_saved.get("ok"):
                                        st.success(f"Saved / updated {ticker} in Candidate Watchlist.")
                                        # Refresh immediately so the Saved Candidates sidebar
                                        # is rebuilt from Supabase and the new ticker appears
                                        # without requiring a manual browser/page refresh.
                                        st.rerun()
                                    else:
                                        st.warning("Candidate watchlist storage is not configured.")
                                except Exception as exc:
                                    st.error(f"Candidate save failed: {exc}")
                        else:
                            st.caption("Candidate Watchlist requires configured durable storage.")

                        export_cand = candidate_summary.copy()
                        export_cand["Candidate Action"] = investigation.get("Candidate Action")
                        export_cand["Action Reason"] = investigation.get("Action Reason")
                        export_cand["Current Mark"] = mark_cand
                        export_cand["Entry Low"] = plan_cand.get("Pullback Entry Low")
                        export_cand["Entry High"] = plan_cand.get("Pullback Entry High")
                        export_cand["Breakout Reference"] = plan_cand.get("Breakout Reference")
                        export_cand["Invalidation Reference"] = plan_cand.get("Invalidation Reference")
                        export_cand["Target 1R"] = plan_cand.get("Target 1R")
                        export_cand["Target 2R"] = plan_cand.get("Target 2R")
                        export_cand["Target 3R"] = plan_cand.get("Target 3R")
                        export_cand["Risk %"] = plan_cand.get("Risk %")
                        export_cand["Recorded At"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

                        st.download_button(
                            "Download Candidate Investigation",
                            export_cand.to_csv(index=False),
                            f"bullseye_phase4q7_candidate_{ticker}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                            "text/csv",
                        )

                except Exception as exc:
                    st.error(
                        f"Candidate investigation failed during **{candidate_stage}**: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    with st.expander("🔧 Candidate diagnostic details"):
                        st.code(traceback.format_exc(), language="text")
                    st.caption(
                        "Phase 4R.2B diagnostic output identifies the exact failing line; "
                        "no Bullseye score or trade-management math has been changed."
                    )

    elif not ticker:
        st.warning("Enter the ticker for the actual position or closed trade.")

    elif phase4q1_entry <= 0:
        st.warning("Enter the actual average entry price per share.")

    elif phase4q1_initial_shares <= 0:
        st.warning("Enter the initial number of shares.")

    else:
        with st.spinner(f"Loading current Bullseye references for {ticker}..."):
            data_q1 = download_prices(sorted(set([ticker, "SPY"])))
            df_q1 = one_symbol(data_q1, ticker)
            spy_q1 = one_symbol(data_q1, "SPY")

        if df_q1 is None or spy_q1 is None:
            st.error("Could not retrieve enough market data for this ticker.")
        else:
            try:
                scored_q1 = score_stock(df_q1, spy_q1)
                plan_q1 = build_trade_plan(df_q1, scored_q1)

                daily_reference_price = float(df_q1["Close"].iloc[-1])
                position_mark = get_position_mark(ticker)

                if position_mark.get("status") == "OK" and pd.notna(position_mark.get("price")):
                    current_q1 = float(position_mark["price"])
                    mark_source_q1 = position_mark["source"]
                    mark_session_q1 = position_mark["session"]
                    mark_timestamp_q1 = position_mark["timestamp"]
                    mark_fallback_q1 = False
                else:
                    current_q1 = daily_reference_price
                    mark_source_q1 = "Bullseye daily close fallback"
                    mark_session_q1 = "Fallback / extended-hours mark unavailable"
                    mark_timestamp_q1 = df_q1.index[-1]
                    mark_fallback_q1 = True

                # Phase 4Q.3 test harness: simulate only the live-position mark.
                # The scanner score, technical plan, and downloaded market data remain untouched.
                if (
                    state == "Entered / Live Position"
                    and phase4q3_test_mode
                    and float(phase4q3_test_mark) > 0
                ):
                    current_q1 = float(phase4q3_test_mark)
                    mark_source_q1 = "Phase 4Q.3 simulated Position Mark"
                    mark_session_q1 = "SIMULATION — not market data"
                    mark_timestamp_q1 = pd.Timestamp.now(tz="America/New_York")
                    mark_fallback_q1 = False

                if plan_q1 is None:
                    st.error("Bullseye could not build a current technical plan for this ticker.")
                else:
                    # -----------------------------
                    # Actual trade inputs
                    # -----------------------------
                    entry = float(phase4q1_entry)
                    initial_shares = float(phase4q1_initial_shares)
                    entered_remaining = float(phase4q1_remaining_shares)
                    remaining = min(max(entered_remaining, 0.0), initial_shares)
                    realized = float(phase4q1_realized_pl)

                    # A zero remaining-share balance is meaningful and must never
                    # be replaced with the original share count.
                    if state == "Closed Trade":
                        effective_state = "Closed Trade"
                        remaining = 0.0
                    elif state == "Entered / Live Position" and remaining <= 0:
                        effective_state = "Closed / No Shares Remaining"
                        remaining = 0.0
                    else:
                        effective_state = state

                    # -----------------------------
                    # Current Bullseye references
                    # -----------------------------
                    bull_entry = (
                        float(plan_q1["Pullback Entry Low"])
                        + float(plan_q1["Pullback Entry High"])
                    ) / 2.0
                    bull_stop = float(plan_q1["Invalidation Reference"])
                    active_stop = (
                        float(phase4q1_actual_stop)
                        if phase4q1_actual_stop > 0
                        else bull_stop
                    )

                    # -----------------------------
                    # Historical/original trade risk
                    # -----------------------------
                    original_stop = (
                        float(phase4q1_initial_stop)
                        if phase4q1_initial_stop > 0
                        else np.nan
                    )
                    valid_original_stop = (
                        pd.notna(original_stop)
                        and original_stop < entry
                    )

                    if valid_original_stop:
                        initial_risk_per_share = entry - original_stop
                        initial_total_risk = initial_risk_per_share * initial_shares
                        actual_t1 = entry + initial_risk_per_share
                        actual_t2 = entry + 2.0 * initial_risk_per_share
                        actual_t3 = entry + 3.0 * initial_risk_per_share
                        trail_trigger = (
                            entry
                            + float(phase4q_trail_start_r)
                            * initial_risk_per_share
                        )
                    else:
                        initial_risk_per_share = np.nan
                        initial_total_risk = np.nan
                        actual_t1 = np.nan
                        actual_t2 = np.nan
                        actual_t3 = np.nan
                        trail_trigger = np.nan

                    # -----------------------------
                    # P/L accounting
                    # -----------------------------
                    unrealized = (
                        (current_q1 - entry) * remaining
                        if effective_state == "Entered / Live Position"
                        and remaining > 0
                        else 0.0
                    )
                    combined_pl = realized + unrealized
                    original_cost = entry * initial_shares
                    combined_return_pct = (
                        combined_pl / original_cost * 100.0
                        if original_cost > 0
                        else np.nan
                    )

                    # Actual R means total trade result divided by the ORIGINAL
                    # dollar risk established when the trade was opened.
                    if (
                        pd.notna(initial_total_risk)
                        and initial_total_risk > 0
                    ):
                        actual_r = combined_pl / initial_total_risk
                    else:
                        actual_r = np.nan

                    phase4q2 = build_live_position_management(
                        entry=entry,
                        mark=current_q1,
                        original_stop=original_stop,
                        raw_current_stop_input=float(phase4q1_actual_stop),
                        active_stop=active_stop,
                        remaining_shares=remaining,
                        bullseye_invalidation=float(plan_q1["Invalidation Reference"]),
                    )

                    phase4q4_state = None
                    if effective_state == "Entered / Live Position" and not phase4q3_test_mode:
                        phase4q4_state = _phase4q4_merge_live_state(
                            ticker=ticker,
                            entry=entry,
                            current_r=phase4q2["current_r"],
                            management_state=phase4q2["state"],
                            management_action=phase4q2["action"],
                            protective_stop=phase4q2["protective_stop"],
                            remaining_shares=remaining,
                            mark=current_q1,
                        )

                    # Phase 4Q.5 durable writes are explicit. 4Q.4 continues to update
                    # the in-session live state, but Supabase is only changed when the
                    # user presses Save / Update Position below.
                    phase4q5_save_result = None

                    # Current open risk only applies to shares that still exist.
                    open_risk = (
                        max(current_q1 - active_stop, 0.0) * remaining
                        if effective_state == "Entered / Live Position"
                        and remaining > 0
                        else 0.0
                    )

                    atr = float(plan_q1["ATR14"])
                    trail_ref = (
                        max(active_stop, current_q1 - float(phase4q_trail_atr) * atr)
                        if effective_state == "Entered / Live Position"
                        and remaining > 0
                        else np.nan
                    )

                    # -----------------------------
                    # Position management state
                    # -----------------------------
                    if effective_state in ("Closed Trade", "Closed / No Shares Remaining"):
                        management_action = "Closed"
                        management_reason = (
                            "No shares remain. Realized P/L is preserved and "
                            "unrealized P/L/open risk are zero."
                        )
                    elif current_q1 <= active_stop:
                        management_action = "Exit / Review"
                        management_reason = "Current price is at or below the active stop reference."
                    elif pd.notna(actual_t2) and current_q1 >= actual_t2:
                        management_action = "Trim / Trail"
                        management_reason = "Position is at 2R or better; protect gains while preserving upside."
                    elif pd.notna(actual_t1) and current_q1 >= actual_t1:
                        management_action = "Trim"
                        management_reason = (
                            f"Position reached at least 1R; consider the configured "
                            f"{int(phase4q_trim_pct)}% partial."
                        )
                    elif pd.notna(trail_trigger) and current_q1 >= trail_trigger:
                        management_action = "Hold / Trail"
                        management_reason = "Position reached the configured trailing threshold."
                    elif current_q1 >= entry:
                        management_action = "Hold"
                        management_reason = "Position is profitable but below the first profit-management threshold."
                    else:
                        management_action = "Hold / Monitor"
                        management_reason = "Position is below entry but remains above the active stop."

                    state_row = pd.DataFrame([{
                        "Ticker": ticker,
                        "Selected State": state,
                        "Effective State": effective_state,
                        "Bullseye Action": scored_q1.get("4I Action"),
                        "Signal Tier": scored_q1.get("4H Signal Tier"),
                        "Bullseye 4.0 Score": scored_q1.get("Bullseye 4.0 Score"),
                        "Actual Entry": round(entry, 2),
                        "Position Mark Price": round(current_q1, 2),
                        "Mark Session": mark_session_q1,
                        "Mark Timestamp": (
                            pd.Timestamp(mark_timestamp_q1).strftime("%Y-%m-%d %H:%M:%S %Z")
                            if mark_timestamp_q1 is not None else "Unavailable"
                        ),
                        "Mark Source": mark_source_q1,
                        "Initial Shares": initial_shares,
                        "Remaining Shares": remaining,
                        "Original Stop at Entry": (
                            round(original_stop, 2)
                            if pd.notna(original_stop)
                            else np.nan
                        ),
                        "Current Bullseye Invalidation": round(bull_stop, 2),
                        "Active Stop": round(active_stop, 2),
                        "Initial Risk / Share": (
                            round(initial_risk_per_share, 2)
                            if pd.notna(initial_risk_per_share)
                            else np.nan
                        ),
                        "Initial Total Risk $": (
                            round(initial_total_risk, 2)
                            if pd.notna(initial_total_risk)
                            else np.nan
                        ),
                        "Actual R (Total Trade)": (
                            round(actual_r, 2)
                            if pd.notna(actual_r)
                            else np.nan
                        ),
                        "Profit Target 1 (+1R)": (
                            round(actual_t1, 2)
                            if pd.notna(actual_t1)
                            else np.nan
                        ),
                        "Profit Target 2 (+2R)": (
                            round(actual_t2, 2)
                            if pd.notna(actual_t2)
                            else np.nan
                        ),
                        "Profit Target 3 (+3R)": (
                            round(actual_t3, 2)
                            if pd.notna(actual_t3)
                            else np.nan
                        ),
                        "Realized P/L $": round(realized, 2),
                        "Unrealized P/L $": round(unrealized, 2),
                        "Combined P/L $": round(combined_pl, 2),
                        "Combined Return %": (
                            round(combined_return_pct, 2)
                            if pd.notna(combined_return_pct)
                            else np.nan
                        ),
                        "Open Risk $": round(open_risk, 2),
                        "Management Action": management_action,
                    }])

                    # Keep the state-row management field aligned with the authoritative live engine
                    # before Section A is rendered.
                    if effective_state == "Entered / Live Position":
                        state_row.loc[:, "Management Action"] = phase4q2["action"]

                    st.markdown("**A. Actual-position state**")

                    # Closed trades should emphasize historical facts, not live-management fields.
                    if effective_state in ("Closed Trade", "Closed / No Shares Remaining"):
                        closed_columns = [
                            "Ticker",
                            "Selected State",
                            "Effective State",
                            "Bullseye Action",
                            "Signal Tier",
                            "Bullseye 4.0 Score",
                            "Actual Entry",
                            "Current Price",
                            "Initial Shares",
                            "Remaining Shares",
                            "Original Stop at Entry",
                            "Initial Risk / Share",
                            "Initial Total Risk $",
                            "Actual R (Total Trade)",
                            "Profit Target 1 (+1R)",
                            "Profit Target 2 (+2R)",
                            "Profit Target 3 (+3R)",
                            "Realized P/L $",
                            "Unrealized P/L $",
                            "Combined P/L $",
                            "Combined Return %",
                            "Open Risk $",
                            "Management Action",
                        ]
                        display_state_row = state_row[
                            [c for c in closed_columns if c in state_row.columns]
                        ]
                    else:
                        display_state_row = state_row

                    st.dataframe(
                        display_state_row,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Initial Shares": st.column_config.NumberColumn(format="%.5f"),
                            "Remaining Shares": st.column_config.NumberColumn(format="%.5f"),
                        },
                    )

                    if phase4q3_test_mode and float(phase4q3_test_mark) > 0:
                        st.warning(
                            "🧪 Phase 4Q.3 TEST MODE — the Position Mark below is simulated. "
                            "Bullseye scoring and downloaded market data are unchanged."
                        )
                    st.markdown("**Position Mark used for actual-position accounting**")
                    mark_cols = st.columns(4)
                    mark_cols[0].metric("Mark Price", f"${current_q1:,.2f}")
                    mark_cols[1].metric("Session", mark_session_q1)
                    mark_cols[2].metric(
                        "Timestamp",
                        pd.Timestamp(mark_timestamp_q1).strftime("%m/%d %H:%M:%S %Z")
                        if mark_timestamp_q1 is not None else "Unavailable",
                    )
                    mark_cols[3].metric("Source", mark_source_q1)

                    if mark_fallback_q1:
                        st.warning(
                            "Extended-hours Position Mark was unavailable, so 4Q.1 is using the latest "
                            "daily close as a fallback. P/L may differ from your broker until a fresher "
                            "mark becomes available."
                        )
                    else:
                        st.caption(
                            "This mark is used only for live position P/L and management. "
                            "Bullseye 4.0 scoring and historical validation remain on the frozen daily-data pipeline."
                        )

                    if state == "Entered / Live Position" and remaining == 0:
                        st.warning(
                            "You selected Entered / Live Position, but Remaining Shares is 0.00000. "
                            "Bullseye is therefore treating this record as closed."
                        )

                    st.markdown("**B. Bullseye reference vs actual trade**")
                    st.caption(
                        "Profit Targets 1–3 are potential sell/trim/exit levels, not additional buy entries. "
                        "They represent +1R, +2R, and +3R from the actual entry when the original stop is known."
                    )
                    compare = pd.DataFrame([
                        {
                            "Measure": "Entry",
                            "Bullseye Reference": round(bull_entry, 2),
                            "Actual Trade": round(entry, 2),
                        },
                        {
                            "Measure": "Original stop when trade opened",
                            "Bullseye Reference": "Historical value not reconstructed",
                            "Actual Trade": (
                                round(original_stop, 2)
                                if pd.notna(original_stop)
                                else "Unknown"
                            ),
                        },
                        {
                            "Measure": "Current Bullseye invalidation (informational only)",
                            "Bullseye Reference": round(bull_stop, 2),
                            "Actual Trade": (
                                round(active_stop, 2)
                                if effective_state == "Entered / Live Position"
                                else "N/A — position closed"
                            ),
                        },
                        {
                            "Measure": "Profit / Exit Target 1 (+1R)",
                            "Bullseye Reference": round(float(plan_q1["Target 1R"]), 2),
                            "Actual Trade": (
                                round(actual_t1, 2)
                                if pd.notna(actual_t1)
                                else "N/A — original stop unknown"
                            ),
                        },
                        {
                            "Measure": "Profit / Exit Target 2 (+2R)",
                            "Bullseye Reference": round(float(plan_q1["Target 2R"]), 2),
                            "Actual Trade": (
                                round(actual_t2, 2)
                                if pd.notna(actual_t2)
                                else "N/A — original stop unknown"
                            ),
                        },
                        {
                            "Measure": "Profit / Exit Target 3 (+3R)",
                            "Bullseye Reference": round(float(plan_q1["Target 3R"]), 2),
                            "Actual Trade": (
                                round(actual_t3, 2)
                                if pd.notna(actual_t3)
                                else "N/A — original stop unknown"
                            ),
                        },
                    ])
                    st.dataframe(compare, use_container_width=True, hide_index=True)

                    if effective_state in ("Closed Trade", "Closed / No Shares Remaining"):
                        st.caption(
                            "Closed-trade note: Current Bullseye references are informational only. "
                            "They describe what Bullseye sees today and did not govern this historical trade. "
                            "The original entry and original stop remain the historical values used for trade-result and R calculations."
                        )

                    st.markdown("**C. Position-management readout**")

                    # For live positions, 4Q.2/4Q.3 is the authoritative management engine.
                    # This keeps every visible management field synchronized.
                    if effective_state == "Entered / Live Position":
                        display_management_action = phase4q2["action"]
                        display_management_reason = phase4q2["reason"]
                    else:
                        display_management_action = management_action
                        display_management_reason = management_reason

                    # Synchronize the already-built state table with the authoritative result.
                    if "Management Action" in state_row.columns:
                        state_row.loc[:, "Management Action"] = display_management_action
                    if "Management Reason" in state_row.columns:
                        state_row.loc[:, "Management Reason"] = display_management_reason

                    st.write(f"**Management:** {display_management_action}")
                    st.write(f"**Reason:** {display_management_reason}")

                    if pd.notna(actual_r):
                        st.write(
                            f"**Actual R (Total Trade):** {actual_r:.2f}R — "
                            "combined realized + unrealized P/L divided by the original dollar risk."
                        )
                    else:
                        st.write(
                            "**Actual R (Total Trade):** N/A — enter the original stop used when "
                            "the trade was opened to calculate historical R accurately."
                        )

                    if effective_state == "Entered / Live Position":
                        st.write(
                            f"**Actual trade P/L:** ${combined_pl:,.2f} "
                            f"({combined_return_pct:.2f}%), including "
                            f"${realized:,.2f} already realized."
                        )
                    else:
                        st.write(
                            f"**Closed-trade result:** ${realized:,.2f} realized, "
                            "$0.00 unrealized, and $0.00 open risk."
                        )

                    if effective_state == "Entered / Live Position":
                        st.divider()
                        st.subheader("🎯 Phase 4Q.2 Live Position Management")
                        q2c = st.columns(4)
                        q2c[0].metric("Management State", phase4q2["state"])
                        q2c[1].metric("Current R", f'{phase4q2["current_r"]:.2f}R' if pd.notna(phase4q2["current_r"]) else "N/A")
                        q2c[2].metric("Position Mark", f"${current_q1:,.2f}")
                        q2c[3].metric("Protective Stop Ref", f'${phase4q2["protective_stop"]:,.2f}' if pd.notna(phase4q2["protective_stop"]) else "N/A")
                        st.caption(f'Protective stop source: {phase4q2["protective_stop_source"]}')
                        st.caption(
                            f'Raw current-stop input: ${phase4q2["raw_current_stop_input"]:,.2f} | '
                            f'4Q.1 resolved active stop: ${phase4q2["resolved_active_stop"]:,.2f}'
                        )
                        st.write(f'**Action:** {phase4q2["action"]}')
                        st.write(f'**Why:** {phase4q2["reason"]}')
                        q2_levels = pd.DataFrame([
                            {"Level":"Actual Entry","Price":entry,"Meaning":"Your actual average fill"},
                            {"Level":"Original Risk Stop","Price": original_stop if pd.notna(original_stop) else np.nan,"Meaning":"Historical stop defining 1R"},
                            {"Level":"+1R Profit / Exit Target","Price":phase4q2["t1"],"Meaning":"First profit-protection threshold"},
                            {"Level":"+2R Profit / Exit Target","Price":phase4q2["t2"],"Meaning":"Partial-profit / trailing threshold"},
                            {"Level":"+3R Profit / Exit Target","Price":phase4q2["t3"],"Meaning":"Winner-protection threshold"},
                            {"Level":"Current Bullseye Breakout Reference","Price":float(plan_q1.get("Breakout Reference", np.nan)),"Meaning":"Current continuation / breakout reference"},
                            {"Level":"Current Bullseye Invalidation","Price":float(plan_q1["Invalidation Reference"]),"Meaning":"Current technical invalidation"},
                            {"Level":"4Q.2 Protective Stop Reference","Price":phase4q2["protective_stop"],"Meaning":f'Management reference from {phase4q2["protective_stop_source"]}; never loosens an established stop automatically'},
                        ])
                        st.dataframe(q2_levels,use_container_width=True,hide_index=True,
                                     column_config={"Price":st.column_config.NumberColumn(format="$%.2f")})
                        st.caption(
                            "4Q.2 is a management overlay only. It does not change Bullseye 4.0 scoring or place orders. "
                            "Protective-stop hierarchy: current user-entered stop → original stop at entry → Bullseye invalidation fallback. "
                            "The overlay will not automatically loosen an established stop."
                        )

                        # Phase 4R.4 observes target approach and favorable excursion without
                        # changing the validated 4Q.2 management state, stop, or action.
                        highest_r_4r4 = (
                            phase4q4_state.get("Highest R", np.nan)
                            if phase4q4_state is not None else phase4q2.get("current_r", np.nan)
                        )
                        phase4r4 = build_phase4r4_profit_protection(
                            entry=entry, mark=current_q1,
                            t1=phase4q2.get("t1", np.nan),
                            t2=phase4q2.get("t2", np.nan),
                            t3=phase4q2.get("t3", np.nan),
                            current_r=phase4q2.get("current_r", np.nan),
                            highest_r=highest_r_4r4,
                        )
                        st.divider()
                        st.subheader("🛡️ Phase 4R.4 Target Approach / Profit Protection")
                        r4c = st.columns(4)
                        r4c[0].metric("4R.4 State", phase4r4["signal"])
                        r4c[1].metric("Progress to +1R", f'{phase4r4["progress_to_t1_pct"]:.0f}%' if pd.notna(phase4r4["progress_to_t1_pct"]) else "N/A")
                        r4c[2].metric("Peak Progress", f'{phase4r4["peak_progress_pct"]:.0f}%' if pd.notna(phase4r4["peak_progress_pct"]) else "N/A")
                        r4c[3].metric("Distance to +1R", f'{phase4r4["distance_to_t1_pct"]:.2f}%' if pd.notna(phase4r4["distance_to_t1_pct"]) else "N/A")
                        st.write(f'**Why:** {phase4r4["why"]}')
                        st.caption("Next trigger: " + str(phase4r4["next_trigger"]))
                        st.caption(
                            "4R.4 is observational only. It does not place orders, change Bullseye 4.0 scoring, "
                            "or override Phase 4Q.2 management/protective-stop logic. The initial 0.75R approach "
                            "and 0.15R pullback observations are forward-validation thresholds, not frozen exit rules."
                        )

                        if phase4q3_test_mode and pd.notna(phase4q2["current_r"]):
                            st.markdown("**🧪 Phase 4Q.3 transition test**")
                            st.write(
                                f"Simulated mark **${current_q1:,.2f}** produces **{phase4q2['current_r']:.2f}R** "
                                f"→ state **{phase4q2['state']}** → action **{phase4q2['action']}**."
                            )
                            q1_q2_match = display_management_action == phase4q2["action"]
                            st.caption(
                                f"Management consistency check: "
                                f"{'PASS' if q1_q2_match else 'FAIL'} — "
                                f"4Q.1 displays '{display_management_action}' and 4Q.2 displays '{phase4q2['action']}'."
                            )
                            if pd.notna(phase4q2.get("current_r_raw")):
                                st.caption(
                                    f"R precision audit: raw {phase4q2['current_r_raw']:.8f}R → "
                                    f"classification {phase4q2['current_r']:.4f}R."
                                )
                            transition_guide = pd.DataFrame([
                                {"Test": "Below protective stop", "Expected State": "Exit", "Expected Stop Behavior": "Exit / review immediately"},
                                {"Test": "-0.5R", "Expected State": "Monitor", "Expected Stop Behavior": "Preserve established stop"},
                                {"Test": "+0.5R", "Expected State": "Hold", "Expected Stop Behavior": "Preserve established stop"},
                                {"Test": "+1.0R", "Expected State": "Protect", "Expected Stop Behavior": "Never below breakeven / established stop"},
                                {"Test": "+2.0R", "Expected State": "Trim", "Expected Stop Behavior": "Never below +1R / established stop"},
                                {"Test": "+3.0R", "Expected State": "Trail", "Expected Stop Behavior": "Never below +2R / established stop"},
                            ])
                            st.dataframe(transition_guide, use_container_width=True, hide_index=True)

                    if effective_state == "Entered / Live Position" and not phase4q3_test_mode:
                        st.divider()
                        st.subheader("🧠 Phase 4Q.4 Persistent Position Management")

                        if phase4q4_state is not None:
                            pcols = st.columns(4)
                            pcols[0].metric(
                                "Highest R Reached",
                                f'{phase4q4_state["Highest R"]:.2f}R'
                                if pd.notna(phase4q4_state.get("Highest R"))
                                else "N/A",
                            )
                            pcols[1].metric("Highest State Reached", phase4q4_state.get("Highest State", "N/A"))
                            pcols[2].metric(
                                "Earned Protective Floor",
                                f'${phase4q4_state["Protective Stop Floor"]:,.2f}'
                                if pd.notna(phase4q4_state.get("Protective Stop Floor"))
                                else "N/A",
                            )
                            pcols[3].metric("Remaining Shares", f'{phase4q4_state.get("Remaining Shares", 0.0):.5f}')

                            st.caption(
                                "4Q.4 remembers live trade progress within this Streamlit session. "
                                "Highest R, highest state, and earned protective floor can move forward but never backward. "
                                "Simulation values are never saved into live state."
                            )

                            earned_floor = phase4q4_state.get("Protective Stop Floor", np.nan)
                            if pd.notna(earned_floor):
                                st.write(f"**4Q.4 ratcheted protective floor:** ${float(earned_floor):,.2f}")

                            if st.button(
                                "Reset saved 4Q.4 state for this position",
                                key=f"phase4q4_reset_{_phase4q4_state_key(ticker, entry)}",
                            ):
                                _phase4q4_clear_state(ticker, entry)
                                st.rerun()

                    if effective_state == "Entered / Live Position" and not phase4q3_test_mode:
                        st.markdown("**🧪 Phase 4Q.4 isolated ratchet test**")
                        st.caption(
                            "This uses a completely separate test memory record and cannot change the real saved live-position state."
                        )

                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("1 — Monitor baseline", key="q44_test_monitor"):
                                _phase4q4_apply_test_step(-0.25, "Monitor", original_stop)
                        with c2:
                            if st.button("2 — +1R Protect", key="q44_test_protect"):
                                _phase4q4_apply_test_step(1.00, "Protect", entry)

                        c3, c4 = st.columns(2)
                        with c3:
                            if st.button("3 — +2R Trim", key="q44_test_trim"):
                                _phase4q4_apply_test_step(2.00, "Trim", entry + (entry - original_stop))
                        with c4:
                            if st.button("4 — +3R Trail", key="q44_test_trail"):
                                _phase4q4_apply_test_step(3.00, "Trail", entry + 2 * (entry - original_stop))

                        c5, c6 = st.columns(2)
                        with c5:
                            if st.button("5 — Pull back to +0.25R", key="q44_test_pullback"):
                                _phase4q4_apply_test_step(0.25, "Hold", original_stop)
                        with c6:
                            if st.button("Reset isolated test", key="q44_test_reset"):
                                _phase4q4_reset_test_state()
                                st.rerun()

                        test_state = st.session_state.get("phase4q4_test_state")
                        if test_state is not None:
                            t1, t2, t3 = st.columns(3)
                            t1.metric("Test Highest R", f'{test_state["Highest R"]:.2f}R')
                            t2.metric("Test Highest State", test_state["Highest State"])
                            floor = test_state["Protective Stop Floor"]
                            t3.metric("Test Protective Floor", f"${floor:,.2f}" if floor != float("-inf") else "N/A")

                            expected_floor = entry + 2 * (entry - original_stop)
                            passed = (
                                test_state["Highest R"] >= 3.0
                                and test_state["Highest State"] == "Trail"
                                and test_state["Protective Stop Floor"] >= expected_floor - 0.01
                            )

                            if passed:
                                st.success(
                                    "4Q.4 ratchet test PASS — the later pullback did not reduce Highest R, Highest State, or the earned protective floor."
                                )
                            else:
                                st.info(
                                    "Run the buttons in order 1 → 2 → 3 → 4 → 5. "
                                    "After step 5, the stored test state should still show +3R, Trail, and the +2R protective floor."
                                )

                    if effective_state in ("Closed Trade", "Closed / No Shares Remaining"):
                        st.divider()
                        st.subheader("🏁 Phase 4Q.9C Close & Archive Trade")
                        st.caption(
                            "Use this for a legitimate completed Bullseye trade. It preserves the trade in the "
                            "durable Closed Trade archive and removes the matching live row from Held Positions "
                            "in one atomic Supabase transaction."
                        )

                        storage_cfg = _phase4q5_storage_config()
                        if not storage_cfg["configured"]:
                            st.warning("Durable storage is not configured, so this trade cannot be archived yet.")
                        elif entry <= 0 or initial_shares <= 0:
                            st.warning(
                                "Load the durable live position first so Bullseye has the original entry and share data."
                            )
                        else:
                            current_position_key = _phase4q5_position_key(ticker, entry)

                            close_c1, close_c2 = st.columns(2)
                            with close_c1:
                                phase4q9_final_exit = st.number_input(
                                    "Final exit price per share ($)",
                                    min_value=0.0,
                                    step=0.01,
                                    format="%.2f",
                                    value=0.0,
                                    key=f"phase4q9_final_exit_{current_position_key}",
                                    help="Enter the actual final execution price from your broker; Bullseye will not substitute the current market mark.",
                                )
                            with close_c2:
                                phase4q9_final_realized = st.number_input(
                                    "Final realized P/L for the full trade ($)",
                                    step=0.01,
                                    format="%.2f",
                                    value=float(realized),
                                    key=f"phase4q9_final_realized_{current_position_key}",
                                    help="Enter the broker-confirmed total realized profit or loss for this completed trade.",
                                )

                            phase4q9_exit_reason = st.selectbox(
                                "Exit reason",
                                [
                                    "",
                                    "Target / profit taking",
                                    "Protective stop",
                                    "Manual profit protection",
                                    "Setup deterioration",
                                    "Time / opportunity rotation",
                                    "Other",
                                ],
                                key=f"phase4q9_exit_reason_{current_position_key}",
                            )
                            phase4q9_notes = st.text_area(
                                "Closeout notes (optional)",
                                key=f"phase4q9_notes_{current_position_key}",
                                placeholder="e.g. Reversed just below T1; manually exited to protect profit.",
                            )

                            close_confirmed = (
                                st.session_state.get("phase4q9_close_confirm_key", "")
                                == current_position_key
                            )

                            if not close_confirmed:
                                if st.button(
                                    "🏁 Close & Archive Trade",
                                    key=f"phase4q9_close_arm_{current_position_key}",
                                    type="primary",
                                    use_container_width=True,
                                    disabled=float(phase4q9_final_exit) <= 0,
                                ):
                                    st.session_state["phase4q9_close_confirm_key"] = current_position_key
                                    st.rerun()
                            else:
                                st.warning(
                                    f"Confirm completion of {ticker}. This will archive the trade and remove it "
                                    "from the live Held Positions list. The archived trade is preserved."
                                )
                                cc1, cc2 = st.columns(2)
                                with cc1:
                                    confirm_close = st.button(
                                        f"Confirm Close {ticker}",
                                        key=f"phase4q9_close_confirm_{current_position_key}",
                                        type="primary",
                                        use_container_width=True,
                                    )
                                with cc2:
                                    cancel_close = st.button(
                                        "Cancel",
                                        key=f"phase4q9_close_cancel_{current_position_key}",
                                        use_container_width=True,
                                    )

                                if cancel_close:
                                    st.session_state["phase4q9_close_confirm_key"] = ""
                                    st.rerun()

                                if confirm_close:
                                    try:
                                        close_result = _phase4q9_close_trade(
                                            ticker=ticker,
                                            entry=entry,
                                            final_exit_price=float(phase4q9_final_exit),
                                            final_realized_pl=float(phase4q9_final_realized),
                                            exit_reason=phase4q9_exit_reason,
                                            notes=phase4q9_notes,
                                        )
                                        if close_result.get("ok"):
                                            try:
                                                live_key = _phase4q4_state_key(ticker, entry)
                                                st.session_state.get("phase4q4_live_state", {}).pop(live_key, None)
                                            except Exception:
                                                pass

                                            st.session_state["phase4q9_message"] = (
                                                f"{ticker} closed and archived successfully. "
                                                "It has been removed from Held Positions."
                                            )
                                            st.session_state["phase4q5_last_message"] = ""
                                            st.session_state["phase4q9_clear_position_on_next_run"] = True
                                            st.rerun()
                                        else:
                                            st.error(
                                                f'Close & Archive failed: {close_result.get("status", "unknown error")}'
                                            )
                                    except Exception as exc:
                                        st.error(f"Close & Archive failed: {exc}")

                            st.caption(
                                "Delete Live Position is still available only for erroneous/test records. "
                                "Legitimate completed trades should use Close & Archive so Bullseye retains the outcome."
                            )

                    if effective_state == "Entered / Live Position" and not phase4q3_test_mode:
                        st.divider()
                        st.subheader("💾 Phase 4Q.5 Durable Position Storage")

                        storage_cfg = _phase4q5_storage_config()
                        if storage_cfg["configured"]:
                            st.success(
                                "Durable storage is configured. Nothing is written to Supabase until you press "
                                "**Save / Update Position**."
                            )

                            save_col, verify_col = st.columns(2)
                            with save_col:
                                save_phase4q5 = st.button(
                                    "💾 Save / Update Position",
                                    key=f"phase4q5_save_{_phase4q5_position_key(ticker, entry)}",
                                    type="primary",
                                    disabled=(
                                        phase4q4_state is None
                                        or effective_state != "Entered / Live Position"
                                        or phase4q3_test_mode
                                    ),
                                )

                            with verify_col:
                                st.caption(
                                    "Save uses an upsert: saving this same position again updates the existing row "
                                    "instead of creating a duplicate."
                                )

                            if save_phase4q5:
                                try:
                                    phase4q5_save_result = _phase4q5_save_position(
                                        ticker=ticker,
                                        position_state=effective_state,
                                        entry=entry,
                                        initial_shares=initial_shares,
                                        remaining_shares=remaining,
                                        realized_pl=realized,
                                        original_stop=original_stop,
                                        current_stop_input=float(phase4q1_actual_stop),
                                        live_state=phase4q4_state,
                                    )
                                    if phase4q5_save_result.get("ok"):
                                        saved_at = pd.Timestamp.now(tz="America/New_York").strftime(
                                            "%Y-%m-%d %H:%M:%S %Z"
                                        )
                                        st.session_state["phase4q5_last_saved"] = {
                                            "ticker": ticker,
                                            "entry": entry,
                                            "saved_at": saved_at,
                                        }
                                        st.session_state["phase4q5_last_message"] = (
                                            f"Saved / updated durable {ticker} position."
                                        )
                                        st.success(
                                            f"Saved / updated {ticker} in durable storage at {saved_at}."
                                        )

                                        promoted_save = (
                                            st.session_state.get("phase4q8_promotion_active", False)
                                            and str(st.session_state.get("phase4q8_promotion_ticker", "")).upper().strip()
                                            == str(ticker).upper().strip()
                                        )

                                        if promoted_save:
                                            # Promotion is complete only after the durable live-position save succeeds.
                                            # Remove the ticker from Saved Candidates automatically so it cannot live
                                            # in both the watchlist and Held Positions at the same time.
                                            try:
                                                _phase4q8_delete_candidate(ticker)
                                                st.session_state["phase4q8_message"] = (
                                                    f"{ticker} promoted to Held Positions and removed from Saved Candidates."
                                                )
                                            except Exception as exc:
                                                st.session_state["phase4q8_message"] = (
                                                    f"{ticker} was saved live, but automatic candidate cleanup failed: {exc}"
                                                )

                                            st.session_state["phase4q8_promotion_active"] = False
                                            st.session_state["phase4q8_promotion_ticker"] = ""
                                            st.session_state["phase4q8_promotion_snapshot"] = {}
                                            st.session_state["phase4q8_clear_selected_candidate_on_next_run"] = True

                                        # Refresh immediately after every successful durable save so the Held
                                        # Positions Dashboard and sidebar lists reflect Supabase without a manual reload.
                                        st.rerun()
                                    else:
                                        st.error(
                                            f'Durable save failed: {phase4q5_save_result.get("status", "unknown error")}'
                                        )
                                except Exception as exc:
                                    phase4q5_save_result = {"ok": False, "status": str(exc)}
                                    st.error(f"Durable save error: {exc}")

                            last_saved = st.session_state.get("phase4q5_last_saved")
                            if last_saved:
                                st.caption(
                                    f'Last durable save: {last_saved["ticker"]} @ {last_saved["saved_at"]}'
                                )

                            st.markdown("---")
                            st.markdown("**Remove erroneous / test live position**")
                            current_position_key = _phase4q5_position_key(ticker, entry)
                            delete_confirmed = (
                                st.session_state.get("phase4q5_delete_confirm_key", "")
                                == current_position_key
                            )

                            if not delete_confirmed:
                                if st.button(
                                    "🗑️ Delete Live Position",
                                    key=f"phase4q5_delete_arm_{current_position_key}",
                                    help="Permanently removes this durable live-position record. Use this for test or erroneous entries, not legitimate completed trades.",
                                ):
                                    st.session_state["phase4q5_delete_confirm_key"] = current_position_key
                                    st.rerun()
                            else:
                                st.warning(
                                    f"This will permanently remove {ticker} from durable Held Positions. "
                                    "It will NOT create a Closed Trade record."
                                )
                                dc1, dc2 = st.columns(2)
                                with dc1:
                                    confirm_delete = st.button(
                                        f"Confirm Delete {ticker}",
                                        key=f"phase4q5_delete_confirm_{current_position_key}",
                                        type="primary",
                                        use_container_width=True,
                                    )
                                with dc2:
                                    cancel_delete = st.button(
                                        "Cancel",
                                        key=f"phase4q5_delete_cancel_{current_position_key}",
                                        use_container_width=True,
                                    )

                                if cancel_delete:
                                    st.session_state["phase4q5_delete_confirm_key"] = ""
                                    st.rerun()

                                if confirm_delete:
                                    try:
                                        _phase4q5_delete_position(ticker, entry)
                                        try:
                                            live_key = _phase4q4_state_key(ticker, entry)
                                            st.session_state.get("phase4q4_live_state", {}).pop(live_key, None)
                                        except Exception:
                                            pass

                                        # Do not mutate widget-bound Phase 4Q.1 keys here:
                                        # those widgets have already been instantiated during this
                                        # Streamlit run. Clearing them here raises StreamlitAPIException.
                                        # The durable row is already deleted; simply close the view
                                        # and rerun so Held Positions is rebuilt from Supabase.
                                        st.session_state["phase4q5_delete_confirm_key"] = ""
                                        st.session_state["phase4q1_view_active"] = False
                                        st.session_state["phase4q5_last_message"] = (
                                            f"Deleted {ticker} from durable Held Positions."
                                        )
                                        st.rerun()
                                    except Exception as exc:
                                        st.error(f"Delete live position failed: {exc}")

                            st.caption(
                                "After a redeploy/restart, enter only the ticker and press **Load saved position**. "
                                "Bullseye will restore the saved trade inputs and seed 4Q.4 with the durable "
                                "highest-R/state/protective-floor record."
                            )
                        else:
                            st.warning(
                                "Durable storage is not configured yet. Bullseye is still using Streamlit session memory only."
                            )
                            st.code(
                                'SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"\n'
                                'SUPABASE_SECRET_KEY = "sb_secret_..."\n'
                                'BULLSEYE_OWNER_ID = "bullseye_primary"',
                                language="toml",
                            )

                    export_q1 = state_row.copy()
                    export_q1["Management Action"] = display_management_action
                    export_q1["Management Reason"] = display_management_reason
                    if phase4q4_state is not None:
                        export_q1["4Q.4 Highest R"] = phase4q4_state.get("Highest R", np.nan)
                        export_q1["4Q.4 Highest State"] = phase4q4_state.get("Highest State", "")
                        export_q1["4Q.4 Protective Floor"] = phase4q4_state.get("Protective Stop Floor", np.nan)
                    export_q1["Recorded At"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.download_button(
                        "Download Phase 4Q.1 position-state record",
                        export_q1.to_csv(index=False),
                        f"bullseye_phase4q1_{ticker}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        "text/csv",
                    )

            except Exception as exc:
                st.error(f"Phase 4Q.1 could not build the position-state record: {exc}")

st.caption(f"Phase 4Q.1 generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.")
