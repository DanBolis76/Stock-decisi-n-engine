# Stock Analyzer V1+
Expanded prototype requested by the user.

## Added modules 1–6
1. Automatic scanner
2. News sentiment / impact prototype
3. Historical backtesting
4. Fidelity CSV portfolio analyzer
5. Alert-rule interface
6. iPhone-friendly Streamlit web UI foundation

## Run
pip install -r requirements.txt
streamlit run app.py

## Important
This is a research prototype. It does not execute trades. News sentiment in V1+ is deliberately simple keyword classification and should be replaced by a production NLP/news service before relying on it. Backtesting is simplified and excludes slippage, commissions, taxes and other real-world effects.
