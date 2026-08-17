import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Bullseye 1–4W", layout="wide")

st.title("🎯 Bullseye 1–4W")
st.caption("Phase 3H — interaction testing for Experimental 3.0, beta, and volatility.")

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
        period="5y",
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

    return {
        "Ticker": None,
        "Score": total,
        "Opportunity Score": opportunity_score,
        "Opportunity Rating": opportunity_label,
        "Experimental 3.0 Score": experimental_score,
        "Experimental 3.0 Rating": experimental_label,
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
                "Experimental 3.0 Score": scored["Experimental 3.0 Score"],
                "Experimental 3.0 Rating": scored["Experimental 3.0 Rating"],
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
                "Bullseye Score": scored["Score"],
                "Opportunity Score": scored["Opportunity Score"],
                "Relative Strength": scored["Relative Strength"],
                "RSI": scored["RSI"],
                "Momentum": scored["Momentum"],
                "Volume": scored["Volume"],
                "Technical": scored["Technical"],
                "Extension Penalty": scored["Extension Penalty"],
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

st.info(
    "Phase 3H keeps Experimental 3.0 frozen and tests whether high beta and/or high volatility improve the performance of top-ranked 3.0 setups at each historical snapshot. "
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
            result = pd.DataFrame(rows).sort_values("Experimental 3.0 Score", ascending=False)
            st.subheader("🏆 Top Bullseye Opportunities")
            st.dataframe(
                result[
                    [
                        "Ticker", "Experimental 3.0 Score", "Experimental 3.0 Rating",
                        "Score", "Opportunity Score", "Opportunity Rating", "Rating", "Price",
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

st.caption(f"Phase 3H generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.")







