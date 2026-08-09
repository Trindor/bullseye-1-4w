
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Bullseye 1–4W", layout="wide")

st.title("🎯 Bullseye 1–4W")
st.caption("Version 1 prototype — ranks liquid U.S. stocks for bullish 1–4 week setups.")

DEFAULT_TICKERS = """
AAPL MSFT NVDA AMZN META GOOGL AVGO AMD TSLA NFLX
JPM V MA XOM CVX COST WMT ORCL CRM PLTR MU INTC
QCOM AMAT LRCX MRVL PANW CRWD UBER HOOD COIN
LLY UNH JNJ ABBV ISRG BSX ABT MDT SYK
""".split()

@st.cache_data(ttl=900)
def download_prices(tickers):
    data = yf.download(
        tickers=tickers,
        period="9mo",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    return data

def one_symbol(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return None
        df = data[ticker].copy()
    else:
        df = data.copy()
    df = df.dropna(subset=["Close", "Volume"])
    return df if len(df) >= 80 else None

def pct(a, b):
    if b == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return (a / b - 1) * 100

def score_stock(df, spy):
    c = df["Close"]
    v = df["Volume"]
    last = c.iloc[-1]

    # 1) Momentum — 20
    r5, r10, r20 = [pct(last, c.iloc[-n-1]) for n in (5,10,20)]
    ma20, ma50, ma200 = c.rolling(20).mean().iloc[-1], c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
    momentum = 0
    momentum += np.clip((r5 + 2) * 1.2, 0, 6)
    momentum += np.clip((r10 + 3) * 0.8, 0, 6)
    momentum += np.clip((r20 + 5) * 0.5, 0, 5)
    momentum += 3 if last > ma20 > ma50 else (2 if last > ma20 else 0)
    momentum = float(np.clip(momentum, 0, 20))

    # 2) Relative volume — 20
    rv20 = v.iloc[-1] / v.tail(20).mean()
    rv5 = v.tail(5).mean() / v.tail(20).mean()
    volume = np.clip((rv20 - 0.8) * 7, 0, 12) + np.clip((rv5 - 0.9) * 8, 0, 8)
    volume = float(np.clip(volume, 0, 20))

    # 3) Relative strength vs SPY — 15
    spyc = spy["Close"]
    rs5 = r10 = np.nan
    stock5 = pct(last, c.iloc[-6])
    stock20 = pct(last, c.iloc[-21])
    spy5 = pct(spyc.iloc[-1], spyc.iloc[-6])
    spy20 = pct(spyc.iloc[-1], spyc.iloc[-21])
    rs5, rs20 = stock5 - spy5, stock20 - spy20
    rs = float(np.clip((rs5 + 2) * 1.8, 0, 7) + np.clip((rs20 + 3) * 1.0, 0, 8))

    # 4) Technical setup — 15
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs14 = gain / loss.replace(0, np.nan)
    rsi = float(100 - 100 / (1 + rs14.iloc[-1])) if pd.notna(rs14.iloc[-1]) else 50
    high20 = c.tail(20).max()
    near_breakout = last >= high20 * 0.98
    macd_fast = c.ewm(span=12, adjust=False).mean()
    macd_slow = c.ewm(span=26, adjust=False).mean()
    macd = macd_fast - macd_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_bull = macd.iloc[-1] > signal.iloc[-1]
    technical = 0
    technical += 6 if near_breakout else 3 if last > ma20 else 0
    technical += 4 if macd_bull else 0
    technical += 3 if 50 <= rsi <= 72 else 1 if 45 <= rsi < 50 else 0
    technical += 2 if last > ma50 else 0
    technical = float(np.clip(technical, 0, 15))

    # 5) Sector strength — placeholder in V1
    sector = 5.0

    # 6) Catalyst — placeholder in V1
    catalyst = 5.0

    # 7) Fundamentals — placeholder in V1
    fundamentals = 2.5

    # 8) Risk/liquidity — 5
    dollar_vol = float((c * v).tail(20).mean())
    risk = 0
    risk += 2 if dollar_vol >= 50_000_000 else 1 if dollar_vol >= 10_000_000 else 0
    vol20 = float(c.pct_change().tail(20).std() * math.sqrt(252) * 100)
    risk += 2 if vol20 < 45 else 1 if vol20 < 70 else 0
    risk += 1 if last > 5 else 0
    risk = float(np.clip(risk, 0, 5))

    total = round(momentum + volume + rs + technical + sector + catalyst + fundamentals + risk, 1)
    label = "Exceptional" if total >= 90 else "Strong" if total >= 80 else "Watch" if total >= 70 else "Weak" if total >= 60 else "Avoid"

    return {
        "Ticker": None, "Score": total, "Rating": label, "Price": round(float(last), 2),
        "5D %": round(float(stock5), 2), "20D %": round(float(stock20), 2),
        "Rel Vol": round(float(rv20), 2), "RS vs SPY 20D": round(float(rs20), 2),
        "RSI": round(rsi, 1), "Momentum": round(momentum, 1),
        "Volume": round(volume, 1), "Relative Strength": round(rs, 1),
        "Technical": round(technical, 1), "Risk/Liquidity": round(risk, 1),
        "Sector": sector, "Catalyst": catalyst, "Fundamentals": fundamentals,
    }

with st.sidebar:
    st.header("Scanner settings")
    universe_text = st.text_area("Tickers (space or newline separated)", " ".join(DEFAULT_TICKERS), height=180)
    tickers = sorted(set(x.upper().strip() for x in universe_text.replace(",", " ").split() if x.strip()))
    run = st.button("🔎 Run scanner", type="primary")

st.info("V1 uses price/volume/relative-strength signals. Sector, catalyst, and fundamental scores are intentionally placeholders until those data feeds are connected.")

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

        result = pd.DataFrame(rows).sort_values("Score", ascending=False)
        st.subheader("Top opportunities")
        if len(result):
            st.dataframe(
                result[["Ticker","Score","Rating","Price","5D %","20D %","Rel Vol","RS vs SPY 20D","RSI",
                        "Momentum","Volume","Relative Strength","Technical","Risk/Liquidity"]],
                use_container_width=True, hide_index=True
            )
            st.download_button("Download results CSV", result.to_csv(index=False), "bullseye_results.csv", "text/csv")
        else:
            st.warning("No usable candidates were returned.")

st.caption(f"Prototype generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.")
