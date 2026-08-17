import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Bullseye 1–4W", layout="wide")

st.title("🎯 Bullseye 1–4W")
st.caption("Phase 4E — diagnose winners vs failures inside high-conviction Bullseye 4.0 setups.")

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
    run_robustness = st.button("🛡️ Run 3I robustness test")
    run_phase4a = st.button("🚀 Run 4A prototype test")
    run_phase4b = st.button("🧪 Run 4B accelerator tuning")
    run_phase4c = st.button("🌐 Run 4C broad-universe test")
    run_phase4d = st.button("🎯 Run 4D threshold test")
    run_phase4e = st.button("🔎 Run 4E high-score diagnostics")

st.info(
    "Phase 4E keeps Bullseye 4.0 frozen and compares winners vs failures inside the 90+, 92.5+, and 95+ high-conviction groups. "
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
                        "Ticker", "Bullseye 4.0 Score", "Bullseye 4.0 Rating", "4.0 Accelerator", "Beta 120D",
                        "Experimental 3.0 Score", "Experimental 3.0 Rating",
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

st.caption(f"Phase 4E generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.")













