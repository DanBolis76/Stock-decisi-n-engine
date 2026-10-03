# Stock Analyzer V1.1
Expanded prototype requested by the user.

## Added modules 1–6
1. Automatic scanner
2. News sentiment / impact prototype
3. Historical backtesting
4. Fidelity CSV portfolio analyzer
5. Alert-rule interface
6. iPhone-friendly Streamlit web UI foundation

## Risk-quality improvements
- Validates that every long setup follows `Stop < Entry Zone < Target`.
- Adds market-quality penalties for small capitalization, low liquidity, high ATR and large overnight gaps.
- Flags earnings expected within seven days when Yahoo Finance supplies the date.
- Shows a conservative gap-adjusted stop-fill estimate instead of implying that stops are guaranteed.
- Adds multi-week backtest windows, configurable slippage, intraday high/low stop checks, maximum drawdown and SPY/buy-and-hold comparisons.

## Run
pip install -r requirements.txt
streamlit run app.py

## Important
This is a research prototype. It does not execute trades. News sentiment in V1+ is deliberately simple keyword classification and should be replaced by a production NLP/news service before relying on it. Backtesting is simplified and excludes slippage, commissions, taxes and other real-world effects.
