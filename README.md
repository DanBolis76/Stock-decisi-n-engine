# Stock Analyzer V1.4
Expanded prototype requested by the user.

## Added modules 1–7
1. Automatic scanner
2. News sentiment / impact prototype
3. Historical backtesting
4. Fidelity CSV portfolio analyzer
5. Manual portfolio with add, update, remove, CSV backup and live analysis
6. Alert-rule interface
7. iPhone-friendly Streamlit web UI foundation

## Risk-quality improvements
- Validates that every long setup follows `Stop < Entry Zone < Target`.
- Adds market-quality penalties for small capitalization, low liquidity, high ATR and large overnight gaps.
- Flags earnings expected within seven days when Yahoo Finance supplies the date.
- Shows a conservative gap-adjusted stop-fill estimate instead of implying that stops are guaranteed.
- Adds multi-week backtest windows, configurable slippage, intraday high/low stop checks, maximum drawdown and SPY/buy-and-hold comparisons.
- Adds an adaptive exit option: 2-ATR initial stop, 3-ATR trailing stop and confirmation from two closes below SMA50. The original fixed 7% stop / 14% target remains available for comparison.
- Adds a transparent Market Strength score using Chaikin Money Flow, six-month performance versus SPY, 30/90-day volume trend and distance from SMA200. Entry and exit adjustments are capped at 10 points.

## Run
pip install -r requirements.txt
streamlit run app.py

## Important
This is a research prototype. It does not execute trades. News sentiment in V1+ is deliberately simple keyword classification and should be replaced by a production NLP/news service before relying on it. Backtesting is simplified and excludes slippage, commissions, taxes and other real-world effects.
