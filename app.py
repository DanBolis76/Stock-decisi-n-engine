
import math, io, re
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
def _legacy_analyze_news(news, ticker="", info=None):
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


def news_intelligence(news, ticker="", info=None):
    """Return one consistent analysis used by both calculations and UI."""
    def contains_term(text, term):
        if " " in term:
            return term in text
        return re.search(rf"\b{re.escape(term)}\b", text) is not None

    info = info or {}
    ticker_lower = ticker.lower().strip()
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
    direct_terms = list(company_keywords.get(ticker.upper(), []))
    for name in [info.get("shortName", ""), info.get("longName", "")]:
        name = str(name).lower().strip()
        if not name:
            continue
        direct_terms.append(name)
        cleaned = re.sub(r"\s+(inc\.?|corporation|corp\.?|ltd\.?|plc)$", "", name)
        if cleaned:
            direct_terms.append(cleaned)
    target_aliases = sorted(
        {alias for alias in [ticker_lower, *direct_terms] if alias},
        key=len,
        reverse=True,
    )
    target_pattern = "(?:" + "|".join(re.escape(alias) for alias in target_aliases) + ")"

    sector_words = [
        "technology", "tech stocks", "big tech", "nasdaq", "semiconductor",
        "semiconductors", "artificial intelligence", "ai stocks",
    ]
    market_words = [
        "s&p 500", "stock market", "wall street", "federal reserve", "fed",
        "interest rates", "inflation", "recession", "economy",
    ]
    positive_phrases = {
        "beats estimates": 5, "raises guidance": 5, "record revenue": 4,
        "price target raised": 4, "analyst upgrade": 4, "wins contract": 4,
        "wins major contract": 4, "fda approves": 5, "fda approval": 5,
        "share buyback": 3, "shrinking share count": 3,
        "regulatory approval": 4, "strong demand": 4,
    }
    negative_phrases = {
        "misses estimates": 5, "cuts guidance": 5, "lowers guidance": 5,
        "price target cut": 4, "analyst downgrade": 4,
        "sec investigation": 5, "class action lawsuit": 4,
        "data breach": 5, "product recall": 5, "profit warning": 5,
        "export restriction": 4, "export restrictions": 4,
        "export ban": 5,
    }
    positive_words = {
        "beat": 3, "beats": 3, "growth": 2, "upgrade": 4, "upgraded": 4,
        "surge": 3, "surges": 3, "record": 2, "strong": 2,
        "raises": 3, "raised": 3, "profit": 2, "profits": 2,
        "bullish": 3, "outperform": 4, "approval": 3, "approved": 3,
        "partnership": 2, "launch": 1, "higher": 2, "rises": 2,
        "rise": 2, "gains": 2, "gain": 2, "boosting": 2, "jumps": 3,
        "rallies": 3, "approves": 3,
    }
    negative_words = {
        "miss": 3, "misses": 3, "downgrade": 4, "downgraded": 4,
        "fall": 2, "falls": 2, "drop": 2, "drops": 2, "lawsuit": 4,
        "cut": 3, "cuts": 3, "weak": 2, "loss": 3, "losses": 3,
        "warning": 3, "bearish": 3, "underperform": 4,
        "investigation": 4, "recall": 4, "layoffs": 3, "lower": 2,
        "slips": 2, "declines": 2, "plunges": 4, "recalled": 5,
        "withdrawn": 4, "restriction": 3, "restrictions": 3,
        "sanctions": 4, "ban": 4, "banned": 4,
    }
    event_words = {
        "Geopolitical/Policy": [
            "trump", "xi", "white house", "congress", "china", "tariff",
            "trade restriction", "trade restrictions", "export control",
            "export controls", "export restriction", "export restrictions",
            "sanction", "sanctions", "export ban", "war",
        ],
        "Earnings": ["earnings", "revenue", "eps", "quarter", "guidance", "profit"],
        "Analyst Rating": ["upgrade", "downgrade", "price target", "outperform", "underperform", "rating"],
        "Management": ["ceo", "cfo", "executive", "resigns", "resigned", "appointed"],
        "Product": ["launch", "iphone", "product", "release", "unveils", "recall", "recalled"],
        "Legal/Regulatory": [
            "lawsuit", "investigation", "regulator", "antitrust", "sec",
            "doj", "fda", "clinical trial",
        ],
        "M&A": ["acquisition", "acquire", "merger", "buyout", "takeover"],
    }
    risk_events = {
        "bankruptcy": (50, "Bankruptcy"), "fraud": (40, "Fraud"),
        "accounting irregularities": (40, "Accounting"),
        "investigation": (25, "Investigation"), "data breach": (25, "Cybersecurity"),
        "recall": (25, "Product recall"), "recalled": (25, "Product recall"),
        "export ban": (20, "Export restriction"),
        "export restriction": (15, "Export restriction"),
        "export restrictions": (15, "Export restriction"),
        "lawsuit": (20, "Legal"),
        "antitrust": (20, "Regulatory"), "ceo resigns": (20, "Management"),
        "layoffs": (12, "Layoffs"), "acquisition": (10, "M&A"),
        "merger": (10, "M&A"), "guidance": (8, "Guidance"),
        "earnings": (8, "Earnings"),
    }
    positive_percent = re.compile(
        r"\b(?:up|gain(?:s|ed)?|jump(?:s|ed)?|surge(?:s|d)?|rise(?:s|n)?|rose|"
        r"climb(?:s|ed)?|soar(?:s|ed)?)\s+(?:more\s+than\s+)?\d+(?:\.\d+)?%"
    )
    negative_percent = re.compile(
        r"\b(?:down|fall(?:s|en)?|fell|drop(?:s|ped)?|decline(?:s|d)?|"
        r"slip(?:s|ped)?|plunge(?:s|d)?|sink(?:s)?|sank)\s+"
        r"(?:more\s+than\s+)?\d+(?:\.\d+)?%"
    )

    stories, total_score, total_weight, risk_scores = [], 0.0, 0.0, []
    seen_titles = set()
    for index, item in enumerate((news or [])[:15]):
        content = item.get("content", item)
        if not isinstance(content, dict):
            continue
        title = content.get("title") or item.get("title") or ""
        summary = content.get("summary") or item.get("summary") or ""
        if not title:
            continue
        lowtitle = title.lower()
        normalized_title = re.sub(r"[^a-z0-9]+", " ", lowtitle).strip()
        if normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)
        analysis_text = f"{title} {title} {summary}".lower()
        is_direct = bool(
            (ticker_lower and ticker_lower in lowtitle)
            or any(term and term in lowtitle for term in direct_terms)
        )
        if is_direct:
            relevance, relevance_weight = "🎯 DIRECT", 1.0
        elif any(word in lowtitle for word in sector_words):
            relevance, relevance_weight = "🏭 SECTOR", 0.20
        elif any(word in lowtitle for word in market_words):
            relevance, relevance_weight = "🌎 MARKET", 0.10
        else:
            continue

        words = set(re.sub(r"[,.:;()]", " ", analysis_text).split())
        positive = sum(w for p, w in positive_phrases.items() if p in analysis_text)
        negative = sum(w for p, w in negative_phrases.items() if p in analysis_text)
        positive += sum(w for token, w in positive_words.items() if token in words)
        negative += sum(w for token, w in negative_words.items() if token in words)
        # Generic words such as "buy" can refer to a competitor. Only score
        # recommendations when the action is explicitly aimed at this ticker.
        if target_aliases:
            target_buy = re.search(
                rf"\b(?:buy|own|accumulate)\b.{{0,24}}(?<!\w){target_pattern}(?!\w)",
                lowtitle,
            )
            target_rejection = re.search(
                rf"\b(?:forget|avoid|sell|dump|skip|ditch)\b.{{0,24}}(?<!\w){target_pattern}(?!\w)",
                lowtitle,
            )
            redirect_to_rivals = re.search(
                rf"\b(?:forget|avoid|sell|skip|ditch)\b.{{0,24}}(?<!\w){target_pattern}(?!\w)"
                rf".{{0,50}}\b(?:buy|own)\b",
                lowtitle,
            )
            if target_buy:
                positive += 3
            if target_rejection:
                negative += 5
            if redirect_to_rivals:
                negative += 1
        if positive_percent.search(analysis_text):
            positive += 4
        if negative_percent.search(analysis_text):
            negative += 4
        score = round(max(-100, min(100, (positive - negative) * 12 * relevance_weight)))
        sentiment = "🟢 Positive" if score >= 15 else "🔴 Negative" if score <= -15 else "🟡 Neutral"

        event_type = "General"
        # The headline expresses the article's main subject. Use the summary
        # only when the headline itself contains no recognizable event.
        for event_text in (lowtitle, analysis_text):
            for event, keywords in event_words.items():
                if any(contains_term(event_text, word) for word in keywords):
                    event_type = event
                    break
            if event_type != "General":
                break
        matched_risks = [
            (points, label)
            for phrase, (points, label) in risk_events.items()
            if contains_term(analysis_text, phrase)
        ]
        event_risk, risk_reason = max(matched_risks, default=(0, "No risk event detected"))
        impact = "HIGH" if event_risk >= 20 or abs(score) >= 60 else "MEDIUM" if event_risk >= 10 or abs(score) >= 30 else "LOW"
        explanation = f"{positive} positive vs {negative} negative signal points"
        recency_weight = max(0.35, 1.0 - index * 0.05)
        total_score += score * recency_weight
        total_weight += recency_weight
        risk_scores.append(event_risk * relevance_weight)
        stories.append({
            "title": title,
            "publisher": content.get("provider", {}).get("displayName") or item.get("publisher") or "Source unavailable",
            "link": content.get("canonicalUrl", {}).get("url") or content.get("clickThroughUrl", {}).get("url") or item.get("link"),
            "relevance": relevance, "score": score, "sentiment": sentiment,
            "event_type": event_type, "impact": impact, "event_risk": round(event_risk * relevance_weight),
            "risk_reason": risk_reason, "explanation": explanation,
        })

    news_score = round(total_score / total_weight) if total_weight else 0
    sorted_risks = sorted(risk_scores, reverse=True)
    news_risk = round(min(100, (sorted_risks[0] if sorted_risks else 0) + sum(sorted_risks[1:]) * 0.25))
    risk_reasons = sorted({s["risk_reason"] for s in stories if s["event_risk"] > 0})
    risk_summary = ", ".join(risk_reasons) if risk_reasons else "No risk event detected"
    return stories, news_score, news_risk, risk_summary


def analyze_news(news, ticker="", info=None):
    _, news_score, news_risk, _ = news_intelligence(news, ticker, info)
    return news_score, news_risk
def plan(t, fs, rs, news_score=0, news_risk=0, asset_type="EQUITY"):
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

    is_etf = str(asset_type).upper() == "ETF"
    if is_etf:
        technical_component = 0.50 * num(t["technical"])
        fundamental_component = 0.0
        risk_component = 0.30 * (100 - rs)
        news_component = 0.20 * ((news_score + 100) / 2)
    else:
        technical_component = 0.45 * num(t["technical"])
        fundamental_component = 0.15 * fs
        risk_component = 0.25 * (100 - rs)
        news_component = 0.15 * ((news_score + 100) / 2)
    raw_entry = clamp(
        technical_component
        + fundamental_component
        + risk_component
        + news_component
    )

    if rr < 1:
        rr_multiplier, rr_label = 0.55, "Avoid"
    elif rr < 1.5:
        rr_multiplier, rr_label = 0.75, "Weak"
    elif rr < 2:
        rr_multiplier, rr_label = 0.90, "Acceptable"
    else:
        rr_multiplier, rr_label = 1.0, "Favorable"

    entry = clamp(raw_entry * rr_multiplier)
    rr_penalty = entry - raw_entry


    exit_score = clamp(
    (15 if p < sma20 else 0)
    + (20 if p < sma50 else 0)
    + (15 if num(t["rrsi"]) > 70 else 0)
    + (15 if num(t["macd"]) < num(t["sig"]) else 0)
    + (15 if rs > 70 else 0)
    + 0.20 * max(0, -news_score)
    + 0.20 * news_risk
)

    score_details = {
        "Technical contribution": technical_component,
        "Fundamental contribution (ETF: not used)" if is_etf else "Fundamental contribution": fundamental_component,
        "Risk contribution": risk_component,
        "News contribution": news_component,
        "Raw Entry Score": raw_entry,
        "R/R penalty": rr_penalty,
        "Final Entry Score": entry,
        "R/R quality": rr_label,
    }

    return entry, exit_score, lo, hi, stop, t1, t2, rr, score_details

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
        news_risk,
        i.get("quoteType", "EQUITY"),
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
    entry,exit,lo,hi,stop,t1,t2,rr,score_details=v
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
        "News Event Risk",
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
        if str(i.get("quoteType", "")).upper() == "ETF":
            st.info("ETF detected: corporate fundamentals are not used in the Entry Score.")
        else:
            st.metric("Fundamental Score",f"{fs:.0f}/100")
        st.dataframe(pd.DataFrame({"Metric":list(fv.keys()),"Value":list(fv.values())}),hide_index=True,use_container_width=True)
    with tabs[2]:
        st.subheader("News Intelligence")
        stories, unified_news_score, unified_news_risk, risk_summary = news_intelligence(
            news, ticker, i
        )

        for story in stories:
            st.markdown(f"### {story['sentiment']} — {story['title']}")
            st.caption(
                f"{story['publisher']} | {story['relevance']} | "
                f"Event: {story['event_type']} | Impact: {story['impact']} | "
                f"Score: {story['score']:+.0f} | Event Risk: {story['event_risk']}/100"
            )
            st.caption(f"Why: {story['explanation']}")
            if story["link"]:
                st.markdown(f"[Open article]({story['link']})")
            st.divider()

        if not stories:
            st.info("No relevant recent headlines were found.")

        st.metric("Overall News Sentiment", f"{unified_news_score:+.0f}/100")
        st.metric("News Event Risk", f"{unified_news_risk:.0f}/100")
        st.caption(
            f"Based on {len(stories)} relevant headlines. "
            f"Risk status: {risk_summary}."
        )
    with tabs[3]:
        st.write({"Entry Zone":f"${lo:.2f}–${hi:.2f}","Stop":f"${stop:.2f}","Target 1":f"${t1:.2f}","Target 2":f"${t2:.2f}","Risk/Reward":f"1:{rr:.1f}"})
        st.subheader("Entry Score Breakdown")
        breakdown_rows = []
        for label, value in score_details.items():
            if label == "R/R quality":
                continue
            breakdown_rows.append({
                "Component": label,
                "Points": round(value, 1),
            })
        st.dataframe(
            pd.DataFrame(breakdown_rows),
            hide_index=True,
            use_container_width=True,
        )
        st.metric("R/R Quality", score_details["R/R quality"])
        if rr < 1:
            st.error("Avoid: expected reward is smaller than the capital at risk.")
        elif rr < 1.5:
            st.warning("Weak setup: Risk/Reward is below 1:1.5.")
        elif rr < 2:
            st.info("Acceptable setup: Risk/Reward is between 1:1.5 and 1:2.")
        else:
            st.success("Favorable setup: Risk/Reward is at least 1:2.")

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
