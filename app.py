
import math, io
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stock Analyzer V1+", page_icon="📈", layout="wide")

# ----------------------------
# Core calculations
# ----------------------------
def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, float(x)))

def rsi(close, period=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def technicals(hist):
    c = hist["Close"].dropna()
    v = hist["Volume"].fillna(0)
    if len(c) < 220: return None
    sma20=c.rolling(20).mean().iloc[-1]; sma50=c.rolling(50).mean().iloc[-1]; sma200=c.rolling(200).mean().iloc[-1]
    ema20=c.ewm(span=20,adjust=False).mean().iloc[-1]
    rrsi=float(rsi(c).iloc[-1])
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    sig=macd.ewm(span=9,adjust=False).mean()
    atr=(hist["High"]-hist["Low"]).rolling(14).mean().iloc[-1]
    p=float(c.iloc[-1]); avgv=v.rolling(20).mean().iloc[-1]; rel=float(v.iloc[-1]/avgv) if avgv else 1
    trend=35 if p>sma200 else 10
    trend+=25 if sma50>sma200 else 5
    trend+=20 if p>sma50 else 5
    trend+=20 if p>ema20 else 5
    mom=50
    mom += 25 if 45<=rrsi<=65 else 10 if 35<=rrsi<45 or 65<rrsi<=72 else -15
    mom += 15 if macd.iloc[-1]>sig.iloc[-1] else -10
    mom += 10 if macd.iloc[-1]>macd.iloc[-2] else -5
    mom=clamp(mom)
    sup=min(float(c.tail(60).min()),float(sma50)); res=max(float(c.tail(60).max()),float(c.tail(252).max()))
    dist=(p-sup)/p if p else 0
    ss=90 if .02<=dist<=.08 else 70 if dist<.12 else 45
    vs=85 if rel>=1.5 else 75 if rel>=1.1 else 40 if rel<.7 else 60
    technical=clamp(.25*trend+.20*mom+.20*ss+.10*vs)
    return locals()

def fundamentals(info):
    keys=["trailingPE","forwardPE","priceToSalesTrailing12Months","returnOnEquity","profitMargins",
          "operatingMargins","revenueGrowth","earningsGrowth","freeCashflow","totalDebt","totalCash","beta"]
    vals={k:info.get(k) for k in keys}
    s=50
    for k, good, bad in [("revenueGrowth",.10,0),("earningsGrowth",.10,0),("returnOnEquity",.15,.08)]:
        x=vals.get(k)
        if x is not None: s += 12 if x>good else 5 if x>bad else -10
    pe=vals.get("forwardPE") or vals.get("trailingPE")
    if pe is not None: s += 8 if pe<20 else 3 if pe<30 else -6
    if vals.get("totalDebt") is not None and vals.get("totalCash") is not None:
        s += 8 if vals["totalCash"]>vals["totalDebt"] else -8
    return clamp(s),vals

def risk(t,info):
    s=35; beta=info.get("beta")
    if beta is not None: s += 15 if beta>1.5 else 7 if beta>1.1 else -5
    atrpct=t["atr"]/t["p"]; s += 18 if atrpct>.05 else 8 if atrpct>.03 else 0
    s += 8 if t["rel"]<.5 else 0
    if info.get("totalDebt") and info.get("totalCash") and info["totalDebt"]>info["totalCash"]: s+=10
    return clamp(s)

def plan(t, fs, rs):
    def num(x):
        if hasattr(x, "iloc"):
            x = x.iloc[-1]
        return float(x)

    p = num(t["p"])
    atr = num(t["atr"])
    sup = num(t["sup"])
    res = num(t["res"])
    sma20 = num(t["sma20"])
    sma50 = num(t["sma50"])

    lo = max(sup, p - 1.5 * atr)
    hi = min(p, p + 0.25 * atr)

    stop = min(sup - 0.5 * atr, p - 2 * atr)
    t1 = max(res, p + 2 * atr)
    t2 = p + 3 * atr

    rr = (t1 - p) / (p - stop) if p > stop else 0

    entry = clamp(
        0.55 * num(t["technical"])
        + 0.15 * fs
        + 0.30 * (100 - rs)
    )

    exit_score = clamp(
        (20 if p < sma20 else 0)
        + (25 if p < sma50 else 0)
        + (20 if num(t["rrsi"]) > 70 else 0)
        + (15 if num(t["macd"]) < num(t["sig"]) else 0)
        + (20 if rs > 70 else 0)
    )

    return entry, exit_score, lo, hi, stop, t1, t2, rr

@st.cache_data(ttl=300)
def get_data(ticker,period="2y"):
    tk=yf.Ticker(ticker)
    return tk.history(period=period,auto_adjust=False), tk.info, tk.news

def analyze(ticker,period="2y"):
    h,i,n=get_data(ticker,period)
    if h.empty: return None
    t=technicals(h)
    if not t: return None
    fs,fv=fundamentals(i); rs=risk(t,i); vals=plan(t,fs,rs)
    return h,i,n,t,fs,fv,rs,vals

# ----------------------------
# Scanner
# ----------------------------
def scanner(tickers, min_entry, max_risk):
    rows=[]
    for sym in tickers:
        try:
            a=analyze(sym.strip().upper(),"1y")
            if not a: continue
            h,i,n,t,fs,fv,rs,v=a
            if v[0]>=min_entry and rs<=max_risk:
                rows.append({"Ticker":sym.upper(),"Price":t["p"],"Entry Score":round(v[0]),"Exit Score":round(v[1]),"Risk":round(rs),
                             "Entry Low":v[2],"Entry High":v[3],"Stop":v[4],"Target 1":v[5],"R/R":round(v[7],2)})
        except Exception: pass
    return pd.DataFrame(rows).sort_values(["Entry Score","R/R"],ascending=False) if rows else pd.DataFrame()

# ----------------------------
# Backtest (rule-based prototype)
# ----------------------------
def backtest(ticker, period="5y", threshold=70):
    h,i,n,t,fs,fv,rs,v=analyze(ticker,period)
    c=h["Close"].copy()
    sma50=c.rolling(50).mean(); sma200=c.rolling(200).mean(); rr=rsi(c)
    position=False; entry=0; trades=[]; equity=1.0; curve=[]
    for idx in range(200,len(c)):
        p=float(c.iloc[idx])
        if not position:
            signal=(p>sma200.iloc[idx] and sma50.iloc[idx]>sma200.iloc[idx] and 45<=rr.iloc[idx]<=65)
            if signal:
                position=True; entry=p
        else:
            stop=entry*.93; target=entry*1.14
            if p<=stop or p>=target or p<sma50.iloc[idx]:
                ret=p/entry-1; equity*=1+ret
                trades.append(ret); position=False
        curve.append((c.index[idx],equity))
    if position:
        ret=float(c.iloc[-1]/entry-1); equity*=1+ret; trades.append(ret)
    eq=pd.DataFrame(curve,columns=["Date","Equity"]).set_index("Date") if curve else pd.DataFrame()
    wins=sum(x>0 for x in trades); ntr=len(trades)
    return {"trades":ntr,"win_rate":(wins/ntr*100 if ntr else 0),"total_return":(equity-1)*100,"equity":eq}

# ----------------------------
# UI
# ----------------------------
st.title("📈 Stock Analyzer V1+")
st.caption("Research dashboard with scanner, news sentiment, backtesting, portfolio CSV analysis and alerts. It does not place trades.")

page=st.sidebar.radio("Module",["Stock Analyzer","Scanner","Backtest","Portfolio","Alerts"])
ticker=st.sidebar.text_input("Ticker","AAPL").upper().strip()

if page=="Stock Analyzer":
    a=analyze(ticker,"2y")
    if not a: st.error("Data unavailable or insufficient history."); st.stop()
    h,i,news,t,fs,fv,rs,v=a
    entry,exit,lo,hi,stop,t1,t2,rr=v
    c1,c2,c3,c4=st.columns(4); c1.metric("Entry Score",f"{entry:.0f}/100"); c2.metric("Exit Score",f"{exit:.0f}/100"); c3.metric("Risk",f"{rs:.0f}/100"); c4.metric("R/R",f"1 : {rr:.1f}")
    st.subheader(f"{ticker} — {i.get('longName',ticker)}")
    st.line_chart(h["Close"].tail(180))
    q1,q2,q3,q4=st.columns(4); q1.metric("Price",f"${t['p']:.2f}"); q2.metric("Entry Zone",f"${lo:.2f}–${hi:.2f}"); q3.metric("Stop",f"${stop:.2f}"); q4.metric("Target 1",f"${t1:.2f}")
    tabs=st.tabs(["Technical","Fundamentals","News","Plan"])
    with tabs[0]:
                technical_data = {
            "Indicator": [
                "SMA20",
                "SMA50",
                "SMA200",
                "RSI14",
                "MACD",
                "Signal",
                "Support",
                "Resistance",
                "Relative Volume",
            ],
            "Value": [
                t["sma20"],
                t["sma50"],
                t["sma200"],
                t["rrsi"],
                float(t["macd"].iloc[-1]),
                float(t["sig"].iloc[-1]),
                t["sup"],
                t["res"],
                t["rel"],
            ],
        }

    st.dataframe(
        pd.DataFrame(technical_data),
        hide_index=True,
        use_container_width=True,
    )
    with tabs[1]:
        st.metric("Fundamental Score",f"{fs:.0f}/100")
        st.dataframe(pd.DataFrame({"Metric":list(fv.keys()),"Value":list(fv.values())}),hide_index=True,use_container_width=True)
    with tabs[2]:
        pos=neg=0
        positive_words=["beat","growth","upgrade","surge","record","strong","buy","raises","positive","profit"]
        negative_words=["miss","downgrade","fall","drop","lawsuit","cut","weak","loss","negative","warning"]
        for n in (news or [])[:15]:
            content = n.get("content", n)

            title = content.get("title") or n.get("title") or "Sin título"

            publisher = (
                content.get("provider", {}).get("displayName")
                or n.get("publisher")
                or "Fuente no disponible"
            )

            link = (
                content.get("canonicalUrl", {}).get("url")
                or content.get("clickThroughUrl", {}).get("url")
                or n.get("link")
            )

            lowtitle = title.lower()

            p = sum(w in lowtitle for w in positive_words)
            ng = sum(w in lowtitle for w in negative_words)

            label = "🟢 Positive" if p > ng else "🔴 Negative" if ng > p else "🟡 Neutral"

            pos += p
            neg += ng

            st.markdown(f"**{label} — {title}**")
            st.caption(publisher)

            if link:
                st.markdown(f"[Open article]({link})")
        st.metric("Headline Sentiment (simple V1)",f"{pos-neg:+d}")
        st.caption("V1 sentiment is keyword-based; a production version should use a dedicated NLP/news API.")
    with tabs[3]:
        st.write({"Entry Zone":f"${lo:.2f}–${hi:.2f}","Stop":f"${stop:.2f}","Target 1":f"${t1:.2f}","Target 2":f"${t2:.2f}","Risk/Reward":f"1:{rr:.1f}"})

elif page=="Scanner":
    st.header("🔎 Opportunity Scanner")
    universe=st.text_area("Tickers (comma separated)","AAPL,MSFT,NVDA,AMZN,GOOGL,META,AVGO,TSLA,AMD,QQQ,SPY")
    min_entry=st.slider("Minimum Entry Score",50,90,70)
    max_risk=st.slider("Maximum Risk Score",20,90,50)
    if st.button("Run Scanner",type="primary"):
        with st.spinner("Analyzing universe…"):
            df=scanner(universe.split(","),min_entry,max_risk)
        st.dataframe(df,hide_index=True,use_container_width=True)
        st.download_button("Download CSV",df.to_csv(index=False),"scanner_results.csv","text/csv")

elif page=="Backtest":
    st.header("🧪 Backtest")
    years=st.selectbox("History",["2y","5y","10y"],index=1)
    if st.button("Run Backtest",type="primary"):
        with st.spinner("Running historical rule test…"):
            b=backtest(ticker,years)
        c1,c2,c3=st.columns(3); c1.metric("Trades",b["trades"]); c2.metric("Win Rate",f"{b['win_rate']:.1f}%"); c3.metric("Total Return",f"{b['total_return']:.1f}%")
        if not b["equity"].empty: st.line_chart(b["equity"])
        st.warning("This is a simplified research backtest, not a live-trading simulation. It excludes slippage, commissions, taxes and survivorship/data-quality effects.")

elif page=="Portfolio":
    st.header("💼 Portfolio Analyzer")
    up=st.file_uploader("Upload Fidelity CSV",type=["csv"])
    if up:
        df=pd.read_csv(up)
        st.dataframe(df,hide_index=True,use_container_width=True)
        # Attempt to locate ticker/symbol column
        cols={c.lower():c for c in df.columns}
        symcol=next((cols[k] for k in ["symbol","ticker"] if k in cols),None)
        if symcol:
            rows=[]
            for sym in df[symcol].dropna().astype(str).str.upper().unique():
                try:
                    a=analyze(sym,"1y")
                    if a:
                        h,i,n,t,fs,fv,rs,v=a
                        rows.append({"Ticker":sym,"Price":t["p"],"Entry Score":round(v[0]),"Exit Score":round(v[1]),"Risk":round(rs),"Fundamental":round(fs),"Entry Low":v[2],"Entry High":v[3],"Stop":v[4],"Target 1":v[5],"R/R":round(v[7],2)})
                except: pass
            if rows: st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.caption("Read-only import. The V1+ never sends orders to Fidelity.")

elif page=="Alerts":
    st.header("🔔 Alert Rules")
    st.write("Create alert definitions for later integration with email/push notifications.")
    sym=st.text_input("Ticker",ticker)
    condition=st.selectbox("Condition",["Entry Score ≥ threshold","Risk Score ≥ threshold","Price ≤ value","Price ≥ value","RSI ≥ value","RSI ≤ value"])
    threshold=st.number_input("Threshold / Value",value=70.0)
    st.code(f"{sym}: {condition} @ {threshold}")
    st.info("The UI stores the rule for this session. Production V2 would connect it to a scheduled server and push/email provider.")

st.sidebar.markdown("---")
st.sidebar.caption("V1+ • Rule-based research prototype • No order execution")
