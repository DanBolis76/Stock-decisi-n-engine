# Stock Decision Engine — iPhone deployment

## What this package is
This is a Streamlit web app. The ZIP does not install as a native iPhone app.
It must be deployed online first. After deployment, open the web URL on iPhone
and use Safari > Share > Add to Home Screen.

## Recommended deployment
Use Streamlit Community Cloud with a GitHub repository.

### 1. GitHub
Create a new repository, for example:
`stock-decision-engine`

Upload these files from this folder:
- `app.py`
- `requirements.txt`
- `README.md`

Do not upload passwords, API keys, Fidelity credentials, or private account data.

### 2. Streamlit
Open https://share.streamlit.io/
Sign in and connect GitHub.
Choose Create app > Yup, I have an app.
Select your repository, branch `main`, and `app.py`.
Deploy.

### 3. iPhone
Open the Streamlit app URL in Safari.
Tap Share -> Add to Home Screen -> Add.
The analyzer will then appear like an app icon.

## Updating the analyzer
When `app.py` or `requirements.txt` changes in GitHub, Streamlit Community
Cloud detects the repository changes and redeploys the app.

## Important
The current prototype is read-only. It does not place trades.
Yahoo/yfinance data is suitable for a prototype but should be replaced with
a licensed market-data provider for production alerts and trading decisions.
