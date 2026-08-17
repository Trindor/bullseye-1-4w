import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Bullseye 1–4W", layout="wide")

st.title("🎯 Bullseye 1–4W")
st.caption("Phase 2F — historical validation and 1–4 week backtesting.")

DEFAULT_TICKERS = """
AAPL MSFT NVDA AMZN META GOOGL AVGO AMD TSLA NFLX
JPM V MA XOM CVX COST WMT ORCL CRM PLTR MU INTC
QCOM AMAT LRCX MRVL PANW CRWD UBER HOOD COIN
LLY UNH JNJ ABBV ISRG BSX ABT MDT SYK
""".split()

@st.cache_data(ttl=900)
def download_prices(tickers):
    return yf.download(
        tickers=tickers,
        period="18mo",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )

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

    return {
        "Ticker": None,
        "Score": total,
        "Opportunity Score": opportunity_score,
        "Opportunity Rating": opportunity_label,
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
                "Setup Quality": scored["Setup Quality"],
                "Extension Penalty": scored["Extension Penalty"],
            }
            for days in (5, 10, 15, 20):
                future = float(df["Close"].iloc[i + days])
                row[f"{days}D Forward %"] = round(pct(future, entry), 2)
            rows.append(row)
        except Exception:
            continue

    return rows

with st.sidebar:
    st.header("Scanner settings")
    universe_text = st.text_area(
        "Tickers (space or newline separated)",
        " ".join(DEFAULT_TICKERS),
        height=180,
    )
    tickers = sorted(
        set(x.upper().strip() for x in universe_text.replace(",", " ").split() if x.strip())
    )
    run = st.button("🔎 Run scanner", type="primary")
    st.divider()
    st.subheader("Historical validation")
    backtest_lookback = st.selectbox(
        "Backtest history",
        [60, 120, 180],
        index=1,
        format_func=lambda x: f"Last {x} trading days",
    )
    backtest_step = st.selectbox(
        "Snapshot frequency",
        [5, 10, 20],
        index=0,
        format_func=lambda x: f"Every {x} trading days",
    )
    run_backtest = st.button("🧪 Run backtest")

st.info(
    "Phase 2F keeps the calibrated scanner and adds historical validation of 1–4 week Opportunity Scores. "
    "The Opportunity Score emphasizes setup quality, relative strength, volume confirmation, momentum, "
    "technical condition, and market regime while penalizing extension risk."
)

if run:
    with st.spinner("Downloading market data and scoring candidates..."):
        tickers2 = sorted(set(tickers + ["SPY"]))
        data = download_prices(tickers2)
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
                    rows.append(row)
                except Exception:
                    continue

        if rows:
            result = pd.DataFrame(rows).sort_values("Opportunity Score", ascending=False)
            st.subheader("🏆 Top Bullseye Opportunities")
            st.dataframe(
                result[
                    [
                        "Ticker", "Score", "Opportunity Score", "Opportunity Rating", "Rating", "Price",
                        "5D %", "20D %", "60D %", "Rel Vol", "RS vs SPY 20D", "RSI",
                        "Momentum", "Volume", "Relative Strength", "Technical",
                        "Setup Quality", "Extension Penalty", "Dist 20MA %", "Momentum Accel",
                        "Market Regime", "Risk/Liquidity",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download results CSV",
                result.to_csv(index=False),
                "bullseye_phase2e_results.csv",
                "text/csv",
            )
        else:
            st.warning("No usable candidates were returned.")


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

            corr_cols = [
                "Opportunity Score", "Setup Quality", "Extension Penalty",
                "5D Forward %", "10D Forward %", "15D Forward %", "20D Forward %"
            ]
            corr = bt[corr_cols].corr(numeric_only=True)["20D Forward %"].drop("20D Forward %")
            corr_df = corr.rename("Correlation with 20D Return").round(3).reset_index()
            corr_df.columns = ["Signal", "Correlation with 20D Return"]

            st.markdown("**Simple signal correlation with 20-day forward return**")
            st.dataframe(corr_df, use_container_width=True, hide_index=True)

            st.markdown("**Historical snapshots**")
            st.dataframe(
                bt.sort_values(["Date", "Opportunity Score"], ascending=[False, False]),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download backtest CSV",
                bt.to_csv(index=False),
                "bullseye_phase2f_backtest.csv",
                "text/csv",
            )
        else:
            st.warning("No historical backtest samples were returned.")

st.caption(f"Phase 2F generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.")

