
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
def analyze_news(news, ticker="", info=None):
    """
    Analyze recent headlines.

    Returns:
        news_score: analyze_news()
        news_risk:   0 to 100
    """

    if not news:
        return 0, 0

    # Strong phrases are checked before individual words.
    positive_phrases = {
        "beats estimates": 5,
        "beats expectations": 5,
        "better than expected": 4,
        "raises guidance": 5,
        "raises outlook": 5,
        "record revenue": 4,
        "record profit": 4,
        "strong earnings": 4,
        "strong demand": 3,
        "price target raised": 4,
        "upgraded to buy": 5,
        "upgraded to outperform": 5,
        "new partnership": 3,
        "strategic partnership": 3,
        "wins contract": 4,
        "major contract": 4,
        "share buyback": 3,
        "stock buyback": 3,
        "dividend increase": 3,
        "regulatory approval": 4,
        "approved by fda": 5,
        "market share gains": 3,
        "expands margins": 3,
        "strong sales": 4,
        "sales growth": 4,
        "revenue growth": 4,
        "profit growth": 4,
        "earnings growth": 4,
        "strong results": 4,
        "better outlook": 4,
        "raises forecast": 5,
        "strong forecast": 4,
        "new product": 2,
        "product launch": 3,
        "market share gains": 4,
        "gains market share": 4,
        "strong demand": 4,
        "analyst upgrade": 4,
        "price target increase": 4,
        "record high": 3,
        "all-time high": 3,
        "new deal": 3,
        "expands partnership": 3,
        
    }

    negative_phrases = {
        "misses estimates": 5,
        "misses expectations": 5,
        "worse than expected": 4,
        "cuts guidance": 5,
        "lowers guidance": 5,
        "cuts outlook": 5,
        "profit warning": 5,
        "revenue decline": 3,
        "sales decline": 3,
        "price target cut": 4,
        "downgraded to sell": 5,
        "downgraded to underperform": 5,
        "sec investigation": 5,
        "doj investigation": 5,
        "antitrust investigation": 5,
        "class action lawsuit": 4,
        "data breach": 5,
        "product recall": 5,
        "files for bankruptcy": 6,
        "bankruptcy filing": 6,
        "accounting irregularities": 6,
        "weak sales": 4,
        "sales slowdown": 4,
        "revenue slowdown": 4,
        "profit decline": 4,
        "earnings decline": 4,
        "weak results": 4,
        "weak outlook": 5,
        "lowers forecast": 5,
        "demand weakness": 4,
        "analyst downgrade": 4,
        "price target lowered": 4,
        "market share loss": 4,
        "loses market share": 4,
        "production delay": 3,
        "product delay": 3,
        "supply shortage": 3,
        "regulatory pressure": 4,"decline": 2,
       
        
    }

    positive_words = {
        "beat": 2,
        "beats": 2,
        "growth": 2,
        "upgrade": 3,
        "upgraded": 3,
        "surge": 2,
        "surges": 2,
        "record": 2,
        "strong": 2,
        "bullish": 2,
        "outperform": 3,
        "approval": 2,
        "approved": 2,
        "partnership": 2,
        "profit": 2,
        "profits": 2,
        "buyback": 2,
        "contract": 2,
        "expansion": 2,
        "guidance": 2,
        "demand": 1,
        "revenue": 1,
        "earnings": 1,
        "margin": 1,
        "margins": 1,
        "contract": 2,
        "expansion": 2,
        "innovation": 1,
        "launch": 1,
        "record": 2,
    }

    negative_words = {
        "miss": 2,
        "misses": 2,
        "downgrade": 3,
        "downgraded": 3,
        "fall": 2,
        "falls": 2,
        "drop": 2,
        "drops": 2,
        "lawsuit": 4,
        "weak": 2,
        "loss": 3,
        "losses": 3,
        "warning": 3,
        "bearish": 2,
        "underperform": 3,
        "investigation": 4,
        "recall": 4,
        "layoffs": 3,
        "bankruptcy": 6,
        "fraud": 6,
        "breach": 4,
        "decline": 2,
        "declines": 2,
        "slowdown": 2,
        "weakness": 2,
        "delay": 2,
        "delays": 2,
        "shortage": 2,
        "pressure": 2,
        "warning": 3,
        "cut": 2,
        "cuts": 2,
        "lowered": 2,
        "fraud": 6,
        "bankruptcy": 6,
        "recall": 4,
        "breach": 4,
    }

    # Events that increase uncertainty/risk even when direction is unclear.
    risk_events = {
        "lawsuit": 20,
        "investigation": 25,
        "sec": 20,
        "doj": 20,
        "antitrust": 20,
        "recall": 25,
        "layoffs": 12,
        "bankruptcy": 50,
        "fraud": 40,
        "data breach": 25,
        "accounting irregularities": 40,
        "ceo resigns": 20,
        "ceo steps down": 20,
        "new ceo": 10,
        "guidance": 8,
        "earnings": 8,
        "acquisition": 10,
        "merger": 10,
    }

    total_score = 0.0
    total_weight = 0.0
    risk_points = 0

    # Analyze up to 15 recent stories.
    for index, item in enumerate((news or [])[:15]):

        content = item.get("content", item)

        if not isinstance(content, dict):
            continue

        title = (
            content.get("title")
            or item.get("title")
            or ""
        )

        summary = (
            content.get("summary")
            or item.get("summary")
            or ""
        )

        if not title:
            continue

        # Analyze both headline and article summary.
        # Headline is repeated so it has more influence than the summary.
        text = f"{title} {title} {summary}".lower().strip()
        # --------------------------------
        # Relevance filter
        ticker_lower = ticker.lower().strip()

        info = info or {}

        company_names = [
            info.get("shortName", ""),
            info.get("longName", ""),
        ]

        # Clean company names and create useful search terms
        company_terms = []

        for name in company_names:
            if name:
                name_lower = str(name).lower().strip()
                company_terms.append(name_lower)

                # Also use the main part of the company name
                for ending in [
                    " inc.",
                    " inc",
                    " corporation",
                    " corp.",
                    " corp",
                    " ltd.",
                    " ltd",
                    " plc",
                ]:
                    if name_lower.endswith(ending):
                        company_terms.append(
                            name_lower[:-len(ending)].strip()
                        )

        ticker_relevant = bool(
            (ticker_lower and ticker_lower in text)
            or any(
                term and term in text
                for term in company_terms
            )
        )
        if ticker_relevant:
            relevance = "🎯 DIRECT"
            relevance_weight = 1.0
        else:
            relevance = "🚫 IRRELEVANT"
            relevance_weight = 0.0
        # Only company-specific stories affect sentiment.
        
        positive = 0
        negative = 0

        # -----------------------------
        # Phrase analysis
        # -----------------------------
        for phrase, weight in positive_phrases.items():
            if phrase in text:
                positive += weight

        for phrase, weight in negative_phrases.items():
            if phrase in text:
                negative += weight

        # -----------------------------
        # Individual word analysis
        # -----------------------------
        words = set(
            text.replace(",", " ")
                .replace(".", " ")
                .replace(":", " ")
                .replace(";", " ")
                .replace("(", " ")
                .replace(")", " ")
                .split()
        )

        for word, weight in positive_words.items():
            if word in words:
                positive += weight

        for word, weight in negative_words.items():
            if word in words:
                negative += weight

        raw_score = positive - negative

        # Convert headline result to -100 ... +100
        headline_score = max(
            -100,
            min(100, raw_score * 12)
        )

        # Newer headlines receive more weight.
        recency_weight = max(0.35, 1.0 - index * 0.05)

        total_score += headline_score * recency_weight * relevance_weight
        total_weight += recency_weight * relevance_weight

        # -----------------------------
        # Event-risk analysis
        # -----------------------------
        headline_risk = 0

        for phrase, points in risk_events.items():
            if phrase in text:
                headline_risk = max(headline_risk, points)

        # Avoid adding unlimited risk for repeated similar stories.
        risk_points += headline_risk * relevance_weight

    # -----------------------------
    # Final news sentiment
    # -----------------------------
    if total_weight > 0:
        average_sentiment = round(total_score / total_weight)
    else:
        average_sentiment = 0

    average_sentiment = max(
        -100,
        min(100, average_sentiment)
    )

    # Risk accumulates more slowly than sentiment.
    news_risk = min(100, round(risk_points * 0.65))

    return average_sentiment, news_risk
def plan(t, fs, rs, news_score=0, news_risk=0):
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
    0.45 * num(t["technical"])
    + 0.15 * fs
    + 0.25 * (100 - rs)
    + 0.15 * ((news_score + 100) / 2)
)


    exit_score = clamp(
    (15 if p < sma20 else 0)
    + (20 if p < sma50 else 0)
    + (15 if num(t["rrsi"]) > 70 else 0)
    + (15 if num(t["macd"]) < num(t["sig"]) else 0)
    + (15 if rs > 70 else 0)
    + 0.20 * max(0, -news_score)
    + 0.20 * news_risk
)

    return entry, exit_score, lo, hi, stop, t1, t2, rr

@st.cache_data(ttl=300)
def get_data(ticker,period="2y"):
    tk=yf.Ticker(ticker)
    return tk.history(period=period,auto_adjust=False), tk.info, tk.news

def analyze(ticker, period="2y"):
    h, i, n = get_data(ticker, period)

    if h.empty:
        return None

    t = technicals(h)

    if not t:
        return None

    fs, fv = fundamentals(i)
    rs = risk(t, i)

    news_score, news_risk = analyze_news(n, ticker, i)

    vals = plan(
        t,
        fs,
        rs,
        news_score,
        news_risk
    )

    return h, i, n, t, fs, fv, rs, vals, news_score, news_risk

# ----------------------------
# Scanner
# ----------------------------
def scanner(tickers, min_entry, max_risk):
    rows=[]
    for sym in tickers:
        try:
            a=analyze(sym.strip().upper(),"1y")
            if not a: continue
            h, i, n, t, fs, fv, rs, v, news_score, news_risk = a
            if v[0]>=min_entry and rs<=max_risk:
                rows.append({"Ticker":sym.upper(),"Price":t["p"],"Entry Score":round(v[0]),"Exit Score":round(v[1]),"Risk":round(rs),
                             "Entry Low":v[2],"Entry High":v[3],"Stop":v[4],"Target 1":v[5],"R/R":round(v[7],2)})
        except Exception: pass
    return pd.DataFrame(rows).sort_values(["Entry Score","R/R"],ascending=False) if rows else pd.DataFrame()

# ----------------------------
# Backtest (rule-based prototype)
# ----------------------------
def backtest(ticker, period="5y", threshold=70):
    h, i, n, t, fs, fv, rs, v, news_score, news_risk = analyze(ticker, period)
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
    h, i, news, t, fs, fv, rs, v, news_score, news_risk = a
    entry,exit,lo,hi,stop,t1,t2,rr=v
    c1,c2,c3,c4=st.columns(4); c1.metric("Entry Score",f"{entry:.0f}/100"); c2.metric("Exit Score",f"{exit:.0f}/100"); c3.metric("Risk",f"{rs:.0f}/100"); c4.metric("R/R",f"1 : {rr:.1f}")
    st.subheader(f"{ticker} — {i.get('longName',ticker)}")
    st.line_chart(h["Close"].tail(180))
    q1,q2,q3,q4=st.columns(4); q1.metric("Price",f"${t['p']:.2f}"); q2.metric("Entry Zone",f"${lo:.2f}–${hi:.2f}"); q3.metric("Stop",f"${stop:.2f}"); q4.metric("Target 1",f"${t1:.2f}")
    n1, n2 = st.columns(2)

    n1.metric(
        "News Score",
        f"{news_score:+.0f}/100"
    )

    n2.metric(
        "News Risk",
        f"{news_risk:.0f}/100"
    )    
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
        st.subheader("News Intelligence")

        positive_words = {
            "beat": 3, "beats": 3,
            "growth": 2,
            "upgrade": 4, "upgraded": 4,
            "surge": 3,
            "record": 2,
            "strong": 2,
            "buy": 3,
            "raises": 3, "raised": 3,
            "profit": 2, "profits": 2,
            "bullish": 3,
            "outperform": 4,
            "approval": 3, "approved": 3,
            "partnership": 2,
            "launch": 1
        }

        negative_words = {
            "miss": 3, "misses": 3,
            "downgrade": 4, "downgraded": 4,
            "fall": 2, "falls": 2,
            "drop": 2, "drops": 2,
            "lawsuit": 4,
            "cut": 3, "cuts": 3,
            "weak": 2,
            "loss": 3, "losses": 3,
            "negative": 2,
            "warning": 3,
            "bearish": 3,
            "underperform": 4,
            "investigation": 4,
            "recall": 4,
            "layoffs": 3
        }

        event_words = {
            "Earnings": [
                "earnings", "revenue", "eps",
                "quarter", "guidance", "profit"
            ],
            "Analyst Rating": [
                "upgrade", "downgrade",
                "price target", "outperform",
                "underperform", "rating"
            ],
            "Management": [
                "ceo", "cfo", "executive",
                "resigns", "resigned", "appointed"
            ],
            "Product": [
                "launch", "iphone", "product",
                "release", "unveils"
            ],
            "Legal/Regulatory": [
                "lawsuit", "investigation",
                "regulator", "antitrust",
                "sec", "doj"
            ],
            "M&A": [
                "acquisition", "acquire",
                "merger", "buyout", "takeover"
            ]
        }

        high_impact_words = [
            "earnings", "guidance", "ceo", "cfo",
            "acquisition", "merger", "lawsuit",
            "investigation", "downgrade",
            "upgrade", "recall"
        ]

        medium_impact_words = [
            "launch", "product", "price target",
            "partnership", "revenue", "profit"
        ]

        relevant_headlines = 0
        company_name = str(i.get("longName") or i.get("shortName") or "").lower()

        ticker_lower = ticker.lower()

        company_keywords = {
            "AAPL": ["apple", "iphone", "ipad", "mac", "tim cook"],
            "MSFT": ["microsoft", "azure", "windows", "satya nadella"],
            "NVDA": ["nvidia", "geforce", "jensen huang"],
            "AMD": ["amd", "advanced micro devices", "lisa su"],
            "AMZN": ["amazon", "aws", "andy jassy"],
            "GOOGL": ["google", "alphabet", "youtube", "gemini"],
            "GOOG": ["google", "alphabet", "youtube", "gemini"],
            "META": ["meta", "facebook", "instagram", "whatsapp", "zuckerberg"],
            "TSLA": ["tesla", "elon musk", "cybertruck"],
        }

        direct_terms = company_keywords.get(ticker.upper(), [])

        if company_name:
            direct_terms.append(company_name)
        for n in (news or [])[:15]:

            content = n.get("content", n)

            title = (
                content.get("title")
                or n.get("title")
                or "Sin título"
            )

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
            # Determine how relevant this headline is to the selected stock
            is_direct = (
                ticker_lower in lowtitle
                or any(term in lowtitle for term in direct_terms)
            )

            sector_words = [
                "technology", "tech stocks", "big tech",
                "nasdaq", "semiconductor", "semiconductors",
                "artificial intelligence", "ai stocks"
            ]

            market_words = [
                "s&p 500", "stock market", "wall street",
                "federal reserve", "fed", "interest rates",
                "inflation", "recession", "economy"
            ]

            is_sector = any(word in lowtitle for word in sector_words)
            is_market = any(word in lowtitle for word in market_words)

            if is_direct:
                relevance = "🎯 DIRECT"
                relevance_weight = 1.0
            elif is_sector:
                relevance = "🏭 SECTOR"
                relevance_weight = 0.20
            elif is_market:
                relevance = "🌎 MARKET"
                relevance_weight = 0.10
            else:
                relevance = "🚫 IRRELEVANT"
                relevance_weight = 0.0
            positive_score = sum(
                weight
                for word, weight in positive_words.items()
                if word in lowtitle
            )

            negative_score = sum(
                weight
                for word, weight in negative_words.items()
                if word in lowtitle
            )

            raw_score = positive_score - negative_score

            sentiment_score = max(
                -100,
                min(100, raw_score * 20 * relevance_weight)
            )

            if relevance_weight > 0:
                relevant_headlines += 1

            if sentiment_score >= 20:
                sentiment = "🟢 Positive"
            elif sentiment_score <= -20:
                sentiment = "🔴 Negative"
            else:
                sentiment = "🟡 Neutral"

            event_type = "General"

            for event, keywords in event_words.items():
                if any(word in lowtitle for word in keywords):
                    event_type = event
                    break

            if any(
                word in lowtitle
                for word in high_impact_words
            ):
                impact = "HIGH"

            elif any(
                word in lowtitle
                for word in medium_impact_words
            ):
                impact = "MEDIUM"

            else:
                impact = "LOW"

            st.markdown(
                f"### {sentiment} — {title}"
            )

            st.caption(
                f"{publisher} | "
                f"{relevance} | "
                f"Event: {event_type} | "
                f"Impact: {impact} | "
                f"Score: {sentiment_score:+.0f}"
            )

            if link:
                st.markdown(
                    f"[Open article]({link})"
                )

            st.divider()

        st.metric(
            "Overall News Sentiment",
            f"{news_score:+.0f}/100"
        )

        st.caption(
            f"Based on {relevant_headlines} relevant headlines. "
            "Positive scores favor bullish sentiment; "
            "negative scores favor bearish sentiment."
        )
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
                        h, i, n, t, fs, fv, rs, v, news_score, news_risk = a
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
