import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Bullseye 1–4W", layout="wide")

st.title("🎯 Bullseye 1–4W")
st.caption("Phase 2B — entry timing + setup quality for bullish 1–4 week opportunities.")

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
    avg20v = v.tail(20).mean()
    avg5v = v.tail(5).mean()
    rv20 = float(v.iloc[-1] / avg20v) if avg20v else 0
    rv5 = float(avg5v / avg20v) if avg20v else 0
    volume = clamp(
        np.clip((rv20 - 0.8) * 5, 0, 8)
        + np.clip((rv5 - 0.9) * 7, 0, 7),
        0, 15
    )

    # 3) Relative strength vs SPY — 15
    spyc = spy["Close"]
    stock5 = pct(last, c.iloc[-6])
    stock20 = pct(last, c.iloc[-21])
    stock60 = pct(last, c.iloc[-61])
    spy5 = pct(spyc.iloc[-1], spyc.iloc[-6])
    spy20 = pct(spyc.iloc[-1], spyc.iloc[-21])
    spy60 = pct(spyc.iloc[-1], spyc.iloc[-61])

    rs5 = stock5 - spy5
    rs20 = stock20 - spy20
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

    total = round(momentum + volume + relative_strength + technical + setup + market_regime + risk, 1)

    if total >= 85:
        label = "Exceptional"
    elif total >= 75:
        label = "Strong"
    elif total >= 65:
        label = "Bullish"
    elif total >= 50:
        label = "Watch"
    else:
        label = "Avoid"

    return {
        "Ticker": None,
        "Score": total,
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
        "Dist 20MA %": round(float(dist20), 2),
        "Momentum Accel": round(float(accel), 2),
        "Market Regime": round(float(market_regime), 1),
        "Risk/Liquidity": round(risk, 1),
    }

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

st.info(
    "Phase 2B adds Setup Quality / Entry Timing. It rewards fresh breakouts, "
    "constructive proximity to moving averages, improving momentum, and healthy volatility expansion, "
    "while penalizing overextended or overheated entries."
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
            result = pd.DataFrame(rows).sort_values("Score", ascending=False)
            st.subheader("🏆 Top Bullseye Opportunities")
            st.dataframe(
                result[
                    [
                        "Ticker", "Score", "Rating", "Price", "5D %", "20D %",
                        "60D %", "Rel Vol", "RS vs SPY 20D", "RSI",
                        "Momentum", "Volume", "Relative Strength",
                        "Technical", "Setup Quality", "Dist 20MA %", "Momentum Accel",
                        "Market Regime", "Risk/Liquidity",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download results CSV",
                result.to_csv(index=False),
                "bullseye_phase2b_results.csv",
                "text/csv",
            )
        else:
            st.warning("No usable candidates were returned.")

st.caption(f"Phase 2B generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.")
