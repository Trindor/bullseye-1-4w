# Bullseye 1–4W — Version 1

This is the first runnable prototype of the 1–4 week bullish stock scanner.

## Run it
1. Install Python 3.10+.
2. In a terminal:
   pip install -r requirements.txt
3. Run:
   streamlit run app.py

## What V1 does
- Downloads daily price/volume data.
- Scores momentum, relative volume, relative strength vs SPY, technical setup, and risk/liquidity.
- Produces a ranked table and CSV export.

## Important
Sector strength, catalysts/news, and fundamentals are placeholders in V1. They are deliberately not fabricated. The next development phase should connect reliable data sources and then build the historical backtester/probability model.

This is a research tool, not a guarantee of future returns or investment advice.
