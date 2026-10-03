
import math, io, re
from datetime import datetime, timezone
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Analyzer V1.5", page_icon="📈", layout="wide")

# ----------------------------
# Core calculations
# ----------------------------
def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, float(x)))

def price_chart(hist, chart_type="Candlestick + Volume", display_period="6 Months", overlays=None):
    """Build an interactive price chart without changing analysis inputs."""
    overlays = overlays or []
    days = {
        "1 Day (5m)": 1,
        "1 Month": 31,
        "3 Months": 93,
        "6 Months": 186,
        "1 Year": 366,
        "2 Years": 732,
    }
    chart_data = hist.copy().dropna(subset=["Open", "High", "Low", "Close"])
    if chart_data.empty:
        return go.Figure()
    cutoff = chart_data.index.max() - pd.Timedelta(days=days[display_period])
    chart_data = chart_data.loc[chart_data.index >= cutoff].copy()

    with_volume = chart_type == "Candlestick + Volume"
    if with_volume:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.76, 0.24],
        )
        price_row = 1
    else:
        fig = go.Figure()
        price_row = None

    trace_args = {"row": price_row, "col": 1} if with_volume else {}
    if chart_type in ("Candlestick", "Candlestick + Volume"):
        fig.add_trace(
            go.Candlestick(
                x=chart_data.index,
                open=chart_data["Open"],
                high=chart_data["High"],
                low=chart_data["Low"],
                close=chart_data["Close"],
                name="OHLC",
                increasing_line_color="#16a34a",
                decreasing_line_color="#dc2626",
            ),
            **trace_args,
        )
    elif chart_type == "OHLC":
        fig.add_trace(
            go.Ohlc(
                x=chart_data.index,
                open=chart_data["Open"],
                high=chart_data["High"],
                low=chart_data["Low"],
                close=chart_data["Close"],
                name="OHLC",
                increasing_line_color="#16a34a",
                decreasing_line_color="#dc2626",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["Close"],
                mode="lines",
                name="Close",
                line={"color": "#2563eb", "width": 2},
                fill="tozeroy" if chart_type == "Area" else None,
                fillcolor="rgba(37, 99, 235, 0.16)",
            )
        )

    for window, color in [(20, "#f59e0b"), (50, "#8b5cf6")]:
        label = f"SMA{window}"
        if label in overlays:
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=hist["Close"].rolling(window).mean().reindex(chart_data.index),
                    mode="lines",
                    name=label,
                    line={"color": color, "width": 1.5},
                ),
                **trace_args,
            )

    if with_volume:
        volume_colors = np.where(
            chart_data["Close"] >= chart_data["Open"],
            "rgba(22, 163, 74, 0.65)",
            "rgba(220, 38, 38, 0.65)",
        )
        fig.add_trace(
            go.Bar(
                x=chart_data.index,
                y=chart_data["Volume"].fillna(0),
                marker_color=volume_colors,
                name="Volume",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.update_layout(
        height=620 if with_volume else 500,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        hovermode="x unified",
        dragmode="zoom",
        legend={"orientation": "h", "y": 1.02, "x": 0},
        xaxis_rangeslider_visible=False,
    )
    if with_volume:
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    else:
        fig.update_yaxes(title_text="Price ($)")
    return fig

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


def market_strength(hist, spy_hist=None):
    """Measure accumulation and relative strength with transparent inputs."""
    data = hist.dropna(subset=["High", "Low", "Close", "Volume"]).copy()
    if len(data) < 200:
        return None

    price_range = (data["High"] - data["Low"]).replace(0, np.nan)
    money_flow_multiplier = (
        ((data["Close"] - data["Low"]) - (data["High"] - data["Close"]))
        / price_range
    ).fillna(0)
    money_flow_volume = money_flow_multiplier * data["Volume"]
    volume_20 = float(data["Volume"].tail(20).sum())
    cmf = float(money_flow_volume.tail(20).sum() / volume_20) if volume_20 else 0.0
    cmf_score = clamp((cmf + 0.20) / 0.40 * 100)

    stock_return = float(data["Close"].iloc[-1] / data["Close"].iloc[-126] - 1)
    spy_return = None
    if spy_hist is not None and not spy_hist.empty:
        spy_close = spy_hist["Close"].dropna()
        if len(spy_close) >= 126:
            spy_return = float(spy_close.iloc[-1] / spy_close.iloc[-126] - 1)
    relative_return = stock_return - spy_return if spy_return is not None else 0.0
    relative_score = clamp((relative_return + 0.20) / 0.40 * 100)

    volume_30 = float(data["Volume"].tail(30).mean())
    volume_90 = float(data["Volume"].tail(90).mean())
    volume_ratio = volume_30 / volume_90 if volume_90 else 1.0
    volume_score = clamp((volume_ratio - 0.60) / 0.80 * 100)

    sma200 = float(data["Close"].rolling(200).mean().iloc[-1])
    price = float(data["Close"].iloc[-1])
    sma200_distance = price / sma200 - 1 if sma200 else 0.0
    trend_score = clamp((sma200_distance + 0.15) / 0.30 * 100)

    components = {
        "Chaikin Money Flow": 0.35 * cmf_score,
        "Relative strength vs SPY": 0.30 * relative_score,
        "Volume trend": 0.20 * volume_score,
        "Position vs SMA200": 0.15 * trend_score,
    }
    score = clamp(sum(components.values()))
    label = (
        "Strong accumulation" if score >= 80
        else "Favorable" if score >= 65
        else "Neutral" if score >= 45
        else "Weak" if score >= 30
        else "Strong distribution"
    )
    return {
        "score": score,
        "label": label,
        "cmf": cmf,
        "stock_return_6m": stock_return,
        "spy_return_6m": spy_return,
        "relative_return_6m": relative_return,
        "volume_ratio": volume_ratio,
        "sma200_distance": sma200_distance,
        "components": components,
    }

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

def market_quality(hist, info):
    """Estimate tradability and event risk from data available before a trade."""
    close = hist["Close"].dropna()
    volume = hist["Volume"].fillna(0)
    avg_volume = float(volume.tail(20).mean()) if not volume.empty else 0.0
    price = float(close.iloc[-1]) if not close.empty else 0.0
    dollar_volume = avg_volume * price
    market_cap = float(info.get("marketCap") or 0)
    previous_close = close.shift(1)
    gaps = ((hist["Open"] / previous_close) - 1).abs().dropna().tail(60)
    gap95 = float(gaps.quantile(0.95)) if not gaps.empty else 0.0
    atr_pct = float(
        ((hist["High"] - hist["Low"]).rolling(14).mean().iloc[-1] / price)
    ) if price else 0.0

    earnings_date = None
    earnings_days = None
    for key in ("earningsTimestampStart", "earningsTimestamp", "earningsTimestampEnd"):
        value = info.get(key)
        if value:
            try:
                earnings_date = datetime.fromtimestamp(float(value), tz=timezone.utc)
                earnings_days = (earnings_date.date() - datetime.now(timezone.utc).date()).days
                break
            except (TypeError, ValueError, OSError):
                pass

    def scaled_score(value, low, high, points):
        if high <= low:
            return 0.0
        return points * clamp((value - low) / (high - low), 0, 1)

    warnings = []
    if market_cap and market_cap < 2_000_000_000:
        warnings.append("Small-cap company (below $2B market value)")
    if avg_volume < 500_000 or dollar_volume < 10_000_000:
        warnings.append("Low liquidity; execution may differ from the displayed price")
    if atr_pct >= 0.05:
        warnings.append("High daily volatility (ATR at least 5% of price)")
    if gap95 >= 0.04:
        warnings.append("Large recent overnight gaps")
    if earnings_days is not None and 0 <= earnings_days <= 7:
        warnings.append(f"Earnings expected in {earnings_days} day(s)")

    # Continuous scores avoid an abrupt jump at a single threshold. Unknown
    # market cap receives a neutral value; unknown earnings do not create an
    # event-risk penalty because there is no verified nearby date.
    volume_score = scaled_score(avg_volume, 100_000, 1_000_000, 15)
    dollar_volume_score = scaled_score(dollar_volume, 2_000_000, 25_000_000, 15)
    liquidity_score = volume_score + dollar_volume_score
    volatility_score = 25 * (1 - clamp((atr_pct - 0.02) / 0.06, 0, 1))
    gap_score = 20 * (1 - clamp((gap95 - 0.01) / 0.05, 0, 1))
    size_score = (
        scaled_score(market_cap, 500_000_000, 10_000_000_000, 10)
        if market_cap
        else 7.0
    )
    if earnings_days is None or earnings_days > 14:
        earnings_score = 15.0
    elif earnings_days < 0:
        earnings_score = 15.0
    else:
        earnings_score = 15 * clamp(earnings_days / 14, 0, 1)

    components = {
        "Liquidity": liquidity_score,
        "Volatility stability": volatility_score,
        "Gap stability": gap_score,
        "Company size": size_score,
        "Earnings distance": earnings_score,
    }
    quality_score = clamp(sum(components.values()))
    penalty = 100 - quality_score
    return {
        "avg_volume": avg_volume,
        "dollar_volume": dollar_volume,
        "market_cap": market_cap,
        "atr_pct": atr_pct,
        "gap95_pct": gap95,
        "earnings_date": earnings_date,
        "earnings_days": earnings_days,
        "penalty": penalty,
        "quality_score": quality_score,
        "components": components,
        "warnings": warnings,
    }


def risk(t,info,quality=None):
    s=35; beta=info.get("beta")
    if beta is not None: s += 15 if beta>1.5 else 7 if beta>1.1 else -5
    atrpct=t["atr"]/t["p"]; s += 18 if atrpct>.05 else 8 if atrpct>.03 else 0
    s += 8 if t["rel"]<.5 else 0
    if info.get("totalDebt") and info.get("totalCash") and info["totalDebt"]>info["totalCash"]: s+=10
    if quality: s += quality["penalty"]
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
def plan(
    t,
    fs,
    rs,
    news_score=0,
    news_risk=0,
    asset_type="EQUITY",
    quality=None,
    strength=None,
):
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
    quality_penalty = float((quality or {}).get("penalty", 0))
    raw_entry_before_quality = clamp(
        technical_component
        + fundamental_component
        + risk_component
        + news_component
    )
    raw_entry = clamp(raw_entry_before_quality - quality_penalty)

    if rr < 1:
        rr_multiplier, rr_label = 0.55, "Avoid"
    elif rr < 1.5:
        rr_multiplier, rr_label = 0.75, "Weak"
    elif rr < 2:
        rr_multiplier, rr_label = 0.90, "Acceptable"
    else:
        rr_multiplier, rr_label = 1.0, "Favorable"

    entry_before_strength = clamp(raw_entry * rr_multiplier)
    rr_penalty = entry_before_strength - raw_entry
    strength_score = float((strength or {}).get("score", 50))
    strength_adjustment = clamp((strength_score - 50) / 5, -10, 10)
    entry = clamp(entry_before_strength + strength_adjustment)


    base_exit_score = clamp(
    (15 if p < sma20 else 0)
    + (20 if p < sma50 else 0)
    + (15 if num(t["rrsi"]) > 70 else 0)
    + (15 if num(t["macd"]) < num(t["sig"]) else 0)
    + (15 if rs > 70 else 0)
    + 0.20 * max(0, -news_score)
    + 0.20 * news_risk
)
    exit_strength_adjustment = clamp((50 - strength_score) / 5, -10, 10)
    exit_score = clamp(base_exit_score + exit_strength_adjustment)

    score_details = {
        "Technical contribution": technical_component,
        "Fundamental contribution (ETF: not used)" if is_etf else "Fundamental contribution": fundamental_component,
        "Risk contribution": risk_component,
        "News contribution": news_component,
        "Market quality penalty": -quality_penalty,
        "Raw Entry Score": raw_entry,
        "R/R penalty": rr_penalty,
        "Market Strength adjustment": strength_adjustment,
        "Final Entry Score": entry,
        "Exit adjustment from Market Strength": exit_strength_adjustment,
        "R/R quality": rr_label,
    }

    return entry, exit_score, lo, hi, stop, t1, t2, rr, score_details

def _google_news_rss(ticker, company_name="", limit=15):
    """Return a Yahoo-compatible news list when Yahoo has no usable headlines."""
    company_name = re.sub(
        r"\s+(inc\.?|corporation|corp\.?|ltd\.?|plc|class [a-z])$",
        "",
        str(company_name).strip(),
        flags=re.IGNORECASE,
    )
    search_name = f'"{company_name}"' if company_name else ticker
    query = quote_plus(f"{search_name} {ticker} stock when:7d")
    url = (
        "https://news.google.com/rss/search?q=" + query
        + "&hl=en-US&gl=US&ceid=US:en"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 StockAnalyzer/1.2"})
    try:
        with urlopen(request, timeout=8) as response:
            root = ET.fromstring(response.read())
    except Exception:
        return []

    stories = []
    for item in root.findall("./channel/item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source = item.find("source")
        publisher = (
            (source.text or "").strip()
            if source is not None
            else "Google News"
        )
        if not title:
            continue
        stories.append({
            "content": {
                "title": title,
                "summary": "",
                "provider": {"displayName": publisher},
                "canonicalUrl": {"url": link},
                "pubDate": (item.findtext("pubDate") or "").strip(),
            }
        })
    return stories


def _deduplicate_news(news):
    unique, seen = [], set()
    for item in news or []:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = str(content.get("title") or item.get("title") or "").strip()
        key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


@st.cache_data(ttl=300)
def get_data(ticker,period="2y"):
    tk = yf.Ticker(ticker)
    history = tk.history(period=period, auto_adjust=False)
    info = tk.info
    news = []
    try:
        news = tk.get_news(count=20, tab="news") or []
    except Exception:
        try:
            news = tk.news or []
        except Exception:
            news = []

    # Yahoo intermittently returns an empty feed. Use an independent
    # company-specific source instead of leaving the News tab blank. Keeping
    # this as a fallback also prevents a scanner run from making one extra
    # network request for every symbol when Yahoo is healthy.
    if not news:
        news = _google_news_rss(
            ticker,
            info.get("shortName") or info.get("longName") or "",
        )
    news = _deduplicate_news(news)
    return history, info, news

@st.cache_data(ttl=60)
def get_intraday_data(ticker):
    """Return the latest trading session in five-minute intervals."""
    return yf.Ticker(ticker).history(
        period="1d",
        interval="5m",
        auto_adjust=False,
        prepost=False,
    )


@st.cache_data(ttl=300)
def get_spy_history(period="2y"):
    """Price-only benchmark fetch shared across analyzer and scanner runs."""
    return yf.Ticker("SPY").history(period=period, auto_adjust=False)

def analyze(ticker, period="2y"):
    h, i, n = get_data(ticker, period)

    if h.empty:
        return None

    t = technicals(h)

    if not t:
        return None

    fs, fv = fundamentals(i)
    quality = market_quality(h, i)
    rs = risk(t, i, quality)
    try:
        strength = market_strength(h, get_spy_history(period))
    except Exception:
        strength = market_strength(h)
    t["market_strength"] = strength

    news_score, news_risk = analyze_news(n, ticker, i)

    vals = plan(
        t,
        fs,
        rs,
        news_score,
        news_risk,
        i.get("quoteType", "EQUITY"),
        quality,
        strength,
    )

    return h, i, n, t, fs, fv, rs, vals, news_score, news_risk, quality

# ----------------------------
# Scanner
# ----------------------------
def scanner(tickers, min_entry, max_risk, min_price=5.0, min_avg_volume=500_000):
    rows=[]
    clean_tickers = list(dict.fromkeys(sym.strip().upper() for sym in tickers if sym.strip()))
    for sym in clean_tickers:
        try:
            a=analyze(sym,"1y")
            if not a: continue
            h, i, n, t, fs, fv, rs, v, news_score, news_risk, quality = a
            close = h["Close"].dropna()
            avg_volume = float(h["Volume"].tail(20).mean())
            day_change = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) > 1 else 0.0
            rel_volume = float(h["Volume"].iloc[-1] / avg_volume) if avg_volume > 0 else 0.0
            if (
                v[0] >= min_entry
                and rs <= max_risk
                and t["p"] >= min_price
                and avg_volume >= min_avg_volume
            ):
                rows.append({
                    "Ticker": sym,
                    "Price": round(t["p"], 2),
                    "Day %": round(day_change, 2),
                    "Rel Volume": round(rel_volume, 2),
                    "Entry Score": round(v[0]),
                    "Risk": round(rs),
                    "R/R": round(v[7], 2),
                    "Exit Score": round(v[1]),
                    "News": round(news_score),
                    "Quality": round(quality["quality_score"]),
                    "Strength": round((t.get("market_strength") or {}).get("score", 50)),
                    "ATR %": round(quality["atr_pct"] * 100, 1),
                    "Gap 95%": round(quality["gap95_pct"] * 100, 1),
                    "Earnings": quality["earnings_date"].strftime("%Y-%m-%d") if quality["earnings_date"] else "Unknown",
                })
        except Exception: pass
    return pd.DataFrame(rows).sort_values(["Entry Score","R/R"],ascending=False) if rows else pd.DataFrame()


@st.cache_data(ttl=300)
def get_most_active_tickers(count=15):
    """Return today's most-active US symbols from Yahoo Finance."""
    response = yf.screen("most_actives", count=count)
    quotes = response.get("quotes", []) if isinstance(response, dict) else []
    return list(dict.fromkeys(
        quote.get("symbol", "").strip().upper()
        for quote in quotes
        if isinstance(quote, dict) and quote.get("symbol")
    ))

# ----------------------------
# Backtest (rule-based prototype)
# ----------------------------
def backtest(ticker, period="5y", test_weeks=26, slippage_bps=10, exit_mode="Adaptive trend"):
    result = analyze(ticker, period)
    if not result:
        return None
    h = result[0].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    c = h["Close"]
    sma50 = c.rolling(50).mean(); sma200 = c.rolling(200).mean(); rr = rsi(c)
    previous_close = c.shift(1)
    true_range = pd.concat([
        h["High"] - h["Low"],
        (h["High"] - previous_close).abs(),
        (h["Low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = true_range.rolling(14).mean()
    start = max(200, len(h) - int(test_weeks * 5.2))
    if start >= len(h) - 1:
        return None
    slip = slippage_bps / 10_000
    position=False; entry=0; trades=[]; equity=1.0; curve=[]
    peak=0; active_stop=0; below_sma50_days=0; exit_reasons={}
    for idx in range(start, len(h)):
        close=float(c.iloc[idx]); high=float(h["High"].iloc[idx]); low=float(h["Low"].iloc[idx])
        if not position:
            signal=(close>sma200.iloc[idx] and sma50.iloc[idx]>sma200.iloc[idx] and 45<=rr.iloc[idx]<=65)
            if signal:
                position=True; entry=close*(1+slip); peak=high; below_sma50_days=0
                entry_atr=float(atr14.iloc[idx]) if pd.notna(atr14.iloc[idx]) else entry*.035
                active_stop=entry-max(2*entry_atr, entry*.05)
        else:
            exit_price=None; exit_reason=None
            if exit_mode == "Adaptive trend":
                # Test today's low against yesterday's stop before using today's high.
                if low<=active_stop:
                    exit_price=active_stop*(1-slip); exit_reason="ATR/trailing stop"
                else:
                    peak=max(peak, high)
                    current_atr=float(atr14.iloc[idx]) if pd.notna(atr14.iloc[idx]) else entry*.035
                    active_stop=max(active_stop, peak-3*current_atr)
                    below_sma50_days = below_sma50_days + 1 if close<sma50.iloc[idx] else 0
                    if below_sma50_days>=2:
                        exit_price=close*(1-slip); exit_reason="Two closes below SMA50"
            else:
                stop=entry*.93; target=entry*1.14
                # Conservative rule: when both occur in one daily bar, the stop wins.
                if low<=stop:
                    exit_price=stop*(1-slip); exit_reason="Fixed stop"
                elif high>=target:
                    exit_price=target*(1-slip); exit_reason="Fixed target"
                elif close<sma50.iloc[idx]:
                    exit_price=close*(1-slip); exit_reason="Close below SMA50"
            if exit_price is not None:
                ret=exit_price/entry-1; equity*=1+ret
                trades.append(ret); position=False
                exit_reasons[exit_reason]=exit_reasons.get(exit_reason,0)+1
        marked_equity = equity * (close / entry) if position else equity
        curve.append((h.index[idx], marked_equity))
    if position:
        exit_price=float(c.iloc[-1])*(1-slip)
        ret=exit_price/entry-1; equity*=1+ret; trades.append(ret)
        exit_reasons["End of test"]=exit_reasons.get("End of test",0)+1
        curve[-1]=(h.index[-1],equity)
    eq=pd.DataFrame(curve,columns=["Date","Strategy"]).set_index("Date") if curve else pd.DataFrame()
    benchmark_return=float(c.iloc[-1]/c.iloc[start]-1)
    spy_return=None
    if not eq.empty:
        eq["Buy & Hold"]=(c.loc[eq.index]/float(c.iloc[start])).values
        try:
            spy_history = get_data("SPY", period)[0]["Close"].dropna()
            spy_aligned = spy_history.reindex(eq.index, method="ffill").dropna()
            if len(spy_aligned) > 1:
                spy_curve = spy_aligned / float(spy_aligned.iloc[0])
                eq["SPY"] = spy_curve.reindex(eq.index, method="ffill")
                spy_return = float((spy_curve.iloc[-1] - 1) * 100)
        except Exception:
            pass
        rolling_peak=eq["Strategy"].cummax()
        max_drawdown=float(((eq["Strategy"]/rolling_peak)-1).min()*100)
    else:
        max_drawdown=0.0
    wins=sum(x>0 for x in trades); ntr=len(trades)
    return {"trades":ntr,"win_rate":(wins/ntr*100 if ntr else 0),
            "total_return":(equity-1)*100,"benchmark_return":benchmark_return*100,
            "spy_return":spy_return,"max_drawdown":max_drawdown,"equity":eq,
            "exit_reasons":exit_reasons,"exit_mode":exit_mode}

# ----------------------------
# UI
# ----------------------------
def status_label(value, green_at, red_below, higher_is_better=True):
    if higher_is_better:
        return "🟢 Favorable" if value >= green_at else "🔴 Unfavorable" if value < red_below else "🟡 Neutral"
    return "🟢 Favorable" if value <= green_at else "🔴 Unfavorable" if value > red_below else "🟡 Neutral"


def style_status_table(frame):
    colors = {
        "🟢": "background-color: #dcfce7; color: #14532d",
        "🟡": "background-color: #fef9c3; color: #713f12",
        "🔴": "background-color: #fee2e2; color: #7f1d1d",
        "🔵": "background-color: #dbeafe; color: #1e3a8a",
    }
    def color_row(row):
        status = str(row.get("Status", ""))
        color = next((css for icon, css in colors.items() if status.startswith(icon)), "")
        return [color] * len(row)
    return frame.style.apply(color_row, axis=1)


def compact_number(value, money=False):
    if value is None or pd.isna(value):
        return "Not available"
    value = float(value)
    prefix = "$" if money else ""
    for divisor, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if abs(value) >= divisor:
            return f"{prefix}{value/divisor:.2f}{suffix}"
    return f"{prefix}{value:.2f}"


st.title("📈 Stock Analyzer V1.5")
st.caption("Research dashboard with scanner, news sentiment, backtesting, portfolio CSV analysis and alerts. It does not place trades.")

page=st.sidebar.radio("Module",["Stock Analyzer","Scanner","Backtest","Portfolio","Alerts"])
ticker=st.sidebar.text_input("Ticker","AAPL").upper().strip()

if page=="Stock Analyzer":
    a=analyze(ticker,"2y")
    if not a: st.error("Data unavailable or insufficient history."); st.stop()
    h, i, news, t, fs, fv, rs, v, news_score, news_risk, quality = a
    entry,exit,lo,hi,stop,t1,t2,rr,score_details=v
    c1,c2,c3,c4=st.columns(4)
    c1.metric(f"{status_label(entry,65,45)} · Entry",f"{entry:.0f}/100",help="Higher is better: attractiveness of a new entry.")
    c2.metric(f"{status_label(exit,30,60,False)} · Exit",f"{exit:.0f}/100",help="Lower is better for holding. A high score means stronger reasons to exit.")
    c3.metric(f"{status_label(rs,40,60,False)} · Risk",f"{rs:.0f}/100",help="Lower is better: estimated market and company risk.")
    c4.metric(f"{status_label(rr,2,1.5)} · R/R",f"1 : {rr:.1f}",help="Potential reward for each $1 at risk. At least 1:2 is preferred.")
    st.subheader(f"{ticker} — {i.get('longName',ticker)}")
    chart_col, period_col, overlay_col = st.columns([1.4, 1, 1.2])
    with chart_col:
        chart_type = st.selectbox(
            "Chart display",
            ["Candlestick + Volume", "Candlestick", "OHLC", "Line", "Area"],
        )
    with period_col:
        display_period = st.selectbox(
            "Display period",
            ["1 Day (5m)", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years"],
            index=3,
        )
    with overlay_col:
        overlays = st.multiselect("Overlays", ["SMA20", "SMA50"], default=["SMA20"])
    chart_history = get_intraday_data(ticker) if display_period == "1 Day (5m)" else h
    if chart_history.empty:
        st.warning("Intraday data is unavailable right now. Showing the daily chart instead.")
        chart_history = h
        display_period = "1 Month"
    st.plotly_chart(
        price_chart(chart_history, chart_type, display_period, overlays),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
    )
    q1,q2,q3,q4=st.columns(4); q1.metric("Price",f"${t['p']:.2f}"); q2.metric("Entry Zone",f"${lo:.2f}–${hi:.2f}"); q3.metric("Stop",f"${stop:.2f}"); q4.metric("Target 1",f"${t1:.2f}")
    n1, n2 = st.columns(2)

    n1.metric(
        f"{status_label(news_score,15,-15)} · News Score",
        f"{news_score:+.0f}/100",
        help="Positive headlines raise the score; negative headlines lower it."
    )

    n2.metric(
        f"{status_label(news_risk,20,50,False)} · News Event Risk",
        f"{news_risk:.0f}/100",
        help="Lower is better. It estimates risk from earnings, legal, regulatory or other news events."
    )
    if quality["warnings"]:
        st.warning("Market quality checks: " + "; ".join(quality["warnings"]) + ".")
    else:
        st.success("Market quality checks: no major liquidity, volatility, gap or near-term earnings warning detected.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{status_label(quality['quality_score'],80,60)} · Market Quality", f"{quality['quality_score']:.0f}/100",help="Higher means easier trading conditions: liquidity, stable volatility, manageable gaps and no nearby earnings.")
    m2.metric(f"{status_label(quality['atr_pct'],.03,.05,False)} · Daily movement", f"{quality['atr_pct']*100:.1f}%",help="ATR as a percentage of price. Lower usually means more stable daily movement.")
    m3.metric(f"{status_label(quality['gap95_pct'],.02,.04,False)} · Overnight gap", f"{quality['gap95_pct']*100:.1f}%",help="A conservative estimate of unusually large overnight price gaps. Lower is better.")
    earnings_days = quality.get("earnings_days")
    earnings_status = (
        "🔵 Unknown" if earnings_days is None
        else "🔴 Near" if 0 <= earnings_days <= 7
        else "🟡 Approaching" if 8 <= earnings_days <= 14
        else "🟢 Not near"
    )
    m4.metric(f"{earnings_status} · Next Earnings", quality["earnings_date"].strftime("%b %d") if quality["earnings_date"] else "Unknown",help="More than 14 days away is favorable; within 7 days adds event risk.")
    component_maximums = {
        "Liquidity": 30,
        "Volatility stability": 25,
        "Gap stability": 20,
        "Company size": 10,
        "Earnings distance": 15,
    }
    st.caption(
        "Market Quality breakdown: "
        + " • ".join(
            f"{name} {score:.1f}/{component_maximums[name]}"
            for name, score in quality["components"].items()
        )
    )
    strength = t.get("market_strength")
    if strength:
        st.subheader("Market Strength")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric(f"{status_label(strength['score'],65,45)} · Market Strength", f"{strength['score']:.0f}/100", strength["label"],help="Higher means stronger accumulation, relative performance, volume trend and long-term price trend.")
        s2.metric(f"{status_label(strength['cmf'],.05,-.05)} · Money Flow", f"{strength['cmf']:+.3f}",help="Above zero suggests buying pressure; below zero suggests selling pressure.")
        s3.metric(
            f"{status_label(strength['relative_return_6m'],.02,-.02)} · 6M vs SPY",
            f"{strength['relative_return_6m']*100:+.1f}%",
            help="How much the stock outperformed or underperformed SPY during six months."
        )
        s4.metric(f"{status_label(strength['volume_ratio'],1,.8)} · Volume trend", f"{strength['volume_ratio']:.2f}x",help="Recent 30-day average volume divided by the 90-day average. Above 1.0 means participation is increasing.")
        strength_maximums = {
            "Chaikin Money Flow": 35,
            "Relative strength vs SPY": 30,
            "Volume trend": 20,
            "Position vs SMA200": 15,
        }
        st.caption(
            "Market Strength breakdown: "
            + " • ".join(
                f"{name} {points:.1f}/{strength_maximums[name]}"
                for name, points in strength["components"].items()
            )
            + ". Entry/Exit adjustments are limited to ±10 points."
        )
    tabs=st.tabs(["Technical","Fundamentals","News","Plan"])
    with tabs[0]:
        current_price = float(t["p"])
        macd_value = float(t["macd"].iloc[-1])
        signal_value = float(t["sig"].iloc[-1])
        rsi_value = float(t["rrsi"])
        relative_volume = float(t["rel"])
        rsi_status = (
            "🟢 Balanced" if 45 <= rsi_value <= 65
            else "🔴 Extended" if rsi_value > 72 or rsi_value < 30
            else "🟡 Watch"
        )
        technical_rows = [
            {"Indicator": "SMA20 · Short trend", "Value": f"${float(t['sma20']):.2f}", "Status": "🟢 Above" if current_price >= t["sma20"] else "🔴 Below", "Plain meaning": "Price compared with its 20-day average."},
            {"Indicator": "SMA50 · Medium trend", "Value": f"${float(t['sma50']):.2f}", "Status": "🟢 Above" if current_price >= t["sma50"] else "🔴 Below", "Plain meaning": "Price compared with its 50-day average."},
            {"Indicator": "SMA200 · Long trend", "Value": f"${float(t['sma200']):.2f}", "Status": "🟢 Above" if current_price >= t["sma200"] else "🔴 Below", "Plain meaning": "Above it generally indicates a positive long-term trend."},
            {"Indicator": "RSI14 · Momentum", "Value": f"{rsi_value:.1f}/100", "Status": rsi_status, "Plain meaning": "45–65 is balanced; above 70 may be overbought and below 30 oversold."},
            {"Indicator": "MACD · Momentum change", "Value": f"{macd_value:+.3f}", "Status": "🟢 Above signal" if macd_value >= signal_value else "🔴 Below signal", "Plain meaning": "Above its signal line suggests improving momentum."},
            {"Indicator": "MACD signal line", "Value": f"{signal_value:+.3f}", "Status": "🔵 Reference", "Plain meaning": "Reference used to interpret MACD."},
            {"Indicator": "Support", "Value": f"${float(t['sup']):.2f}", "Status": "🔵 Reference", "Plain meaning": "Recent area where buyers previously supported the price."},
            {"Indicator": "Resistance", "Value": f"${float(t['res']):.2f}", "Status": "🔵 Reference", "Plain meaning": "Recent area where selling previously limited the price."},
            {"Indicator": "Relative volume", "Value": f"{relative_volume:.2f}x", "Status": status_label(relative_volume,1.1,.7), "Plain meaning": "1.00x equals normal volume; above 1.10x shows stronger participation."},
        ]
        st.dataframe(
            style_status_table(pd.DataFrame(technical_rows)),
            hide_index=True,
            use_container_width=True,
        )
    with tabs[1]:
        if str(i.get("quoteType", "")).upper() == "ETF":
            st.info("ETF detected: corporate fundamentals are not used in the Entry Score.")
        else:
            st.metric(f"{status_label(fs,65,45)} · Fundamental Score",f"{fs:.0f}/100",help="Higher means stronger growth, profitability, valuation and balance-sheet characteristics.")
        fundamental_rules = {
            "trailingPE": ("Trailing P/E", "Price paid for each $1 of past earnings."),
            "forwardPE": ("Forward P/E", "Price paid for each estimated $1 of future earnings."),
            "priceToSalesTrailing12Months": ("Price / Sales", "Price relative to the company's annual revenue."),
            "returnOnEquity": ("Return on equity", "Profit generated from shareholder capital."),
            "profitMargins": ("Profit margin", "Profit retained from each dollar of sales."),
            "operatingMargins": ("Operating margin", "Operating profit from each dollar of sales."),
            "revenueGrowth": ("Revenue growth", "Recent change in company sales."),
            "earningsGrowth": ("Earnings growth", "Recent change in company profits."),
            "freeCashflow": ("Free cash flow", "Cash remaining after operating and investment expenses."),
            "totalDebt": ("Total debt", "All reported company debt."),
            "totalCash": ("Total cash", "Cash and similar liquid resources."),
            "beta": ("Beta", "Price sensitivity versus the market; 1.0 is similar to the market."),
        }
        percent_metrics = {"returnOnEquity", "profitMargins", "operatingMargins", "revenueGrowth", "earningsGrowth"}
        money_metrics = {"freeCashflow", "totalDebt", "totalCash"}
        fundamental_rows = []
        for key, value in fv.items():
            label, meaning = fundamental_rules.get(key, (key, "Company financial metric."))
            if value is None or pd.isna(value):
                display, status = "Not available", "🟡 Missing data"
            elif key in percent_metrics:
                display = f"{float(value)*100:+.1f}%"
                status = status_label(float(value), .10, 0)
            elif key in money_metrics:
                display = compact_number(value, money=True)
                if key == "freeCashflow":
                    status = "🟢 Positive" if value > 0 else "🔴 Negative"
                elif key == "totalCash":
                    status = "🟢 Above debt" if value > (fv.get("totalDebt") or 0) else "🟡 Below debt"
                else:
                    status = "🟢 Below cash" if value < (fv.get("totalCash") or 0) else "🔴 Above cash"
            elif key in {"trailingPE", "forwardPE"}:
                display = f"{float(value):.1f}x"
                status = "🟢 Reasonable" if 0 < value <= 25 else "🟡 Elevated" if 25 < value <= 40 else "🔴 High/invalid"
            elif key == "priceToSalesTrailing12Months":
                display = f"{float(value):.1f}x"
                status = "🟢 Low" if value <= 3 else "🟡 Medium" if value <= 6 else "🔴 High"
            elif key == "beta":
                display = f"{float(value):.2f}"
                status = status_label(float(value), 1.1, 1.5, False)
            else:
                display, status = f"{float(value):.2f}", "🔵 Reference"
            fundamental_rows.append({"Metric": label, "Value": display, "Status": status, "Plain meaning": meaning})
        st.dataframe(style_status_table(pd.DataFrame(fundamental_rows)),hide_index=True,use_container_width=True)
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
        if not (stop < lo <= hi < t1):
            st.error("Invalid trade plan: the required order is Stop < Entry Zone < Target. Do not use this setup.")
        else:
            estimated_gap_stop = stop * (1 - quality["gap95_pct"])
            st.caption(
                f"Stop check: ${stop:.2f} is below the entry zone. A stop is not guaranteed; "
                f"with the recent 95th-percentile gap, a conservative fill could be near ${estimated_gap_stop:.2f}."
            )
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
    core_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "AMD", "QQQ", "SPY"]
    scanner_mode = st.radio(
        "Scanner universe",
        ["Core + My Watchlist", "Most Active Today", "Core list", "My Watchlist"],
        horizontal=True,
    )
    watchlist_text = st.text_area(
        "My watchlist (comma separated)",
        "COST,JPM,LLY,UNH",
        disabled=scanner_mode in ("Core list", "Most Active Today"),
    )
    watchlist = [sym.strip().upper() for sym in watchlist_text.split(",") if sym.strip()]
    if scanner_mode == "Core list":
        universe = core_tickers
    elif scanner_mode == "My Watchlist":
        universe = watchlist
    elif scanner_mode == "Most Active Today":
        universe = []
    else:
        universe = core_tickers + watchlist

    if scanner_mode == "Most Active Today":
        active_count = st.slider("Number of most-active stocks", 5, 25, 15, 5)
        st.caption("The list is refreshed from Yahoo Finance every 5 minutes and then evaluated with the filters below.")
    else:
        active_count = 15
        st.caption(f"Scanning {len(set(universe))} unique symbols. Duplicates are removed automatically.")
    f1, f2 = st.columns(2)
    min_entry = f1.slider("Minimum Entry Score", 0, 90, 50)
    max_risk = f2.slider("Maximum Risk Score", 20, 100, 60)
    q1, q2 = st.columns(2)
    min_price = q1.number_input("Minimum price ($)", min_value=0.0, value=5.0, step=1.0)
    min_avg_volume = q2.number_input("Minimum 20-day average volume", min_value=0, value=500_000, step=100_000)
    if st.button("Run Scanner",type="primary"):
        if scanner_mode == "Most Active Today":
            try:
                universe = get_most_active_tickers(active_count)
            except Exception as exc:
                universe = []
                st.error(f"The most-active list is temporarily unavailable: {exc}")
            if universe:
                st.caption("Most active symbols: " + ", ".join(universe))
        with st.spinner("Analyzing universe…"):
            df=scanner(universe,min_entry,max_risk,min_price,min_avg_volume)
        if df.empty:
            st.info("No stocks matched every filter. Try lowering Entry Score or increasing Maximum Risk.")
        else:
            st.success(f"{len(df)} opportunities matched all filters.")
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Day %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Rel Volume": st.column_config.NumberColumn(format="%.2fx"),
                    "R/R": st.column_config.NumberColumn(format="1 : %.2f"),
                },
            )
            st.download_button("Download CSV",df.to_csv(index=False),"scanner_results.csv","text/csv")

elif page=="Backtest":
    st.header("🧪 Backtest")
    years=st.selectbox("History",["2y","5y","10y"],index=1)
    test_weeks=st.selectbox("Test window",[4,8,12,26,52,104],index=3,format_func=lambda x: f"{x} weeks")
    exit_mode=st.selectbox("Exit strategy",["Adaptive trend","Fixed 7% / 14%"])
    slippage_bps=st.number_input("Slippage per entry/exit (basis points)",min_value=0,max_value=100,value=10,step=5)
    if st.button("Run Backtest",type="primary"):
        with st.spinner("Running historical rule test…"):
            b=backtest(ticker,years,test_weeks,slippage_bps,exit_mode)
        if not b:
            st.error("There is not enough price history for this test window.")
        else:
            c1,c2,c3,c4,c5=st.columns(5)
            c1.metric("Trades",b["trades"]); c2.metric("Win Rate",f"{b['win_rate']:.1f}%")
            c3.metric("Strategy",f"{b['total_return']:.1f}%")
            c4.metric(f"{ticker} Buy & Hold",f"{b['benchmark_return']:.1f}%")
            c5.metric("Max Drawdown",f"{b['max_drawdown']:.1f}%")
            if b["spy_return"] is not None:
                st.metric("SPY reference",f"{b['spy_return']:.1f}%")
            st.caption("Exit reasons: " + ", ".join(f"{name}: {count}" for name,count in b["exit_reasons"].items()))
            if not b["equity"].empty: st.line_chart(b["equity"])
            st.warning("Research simulation only. The adaptive exit uses a 2-ATR initial stop, a 3-ATR trailing stop and two closes below SMA50. It models slippage but still excludes taxes, commissions and exact intraday sequencing.")

elif page=="Portfolio":
    st.header("💼 Portfolio Analyzer")

    if "manual_portfolio" not in st.session_state:
        st.session_state.manual_portfolio = []

    st.subheader("Manual Portfolio")
    st.caption(
        "Add a position manually. Saving an existing ticker updates it. "
        "Download the CSV to preserve the portfolio between deployments or browser sessions."
    )
    with st.form("manual_position_form", clear_on_submit=True):
        p1, p2, p3 = st.columns(3)
        manual_ticker = p1.text_input("Ticker", placeholder="AAPL").strip().upper()
        manual_shares = p2.number_input(
            "Shares",
            min_value=0.000001,
            value=1.0,
            step=0.1,
            format="%.6f",
        )
        manual_cost = p3.number_input(
            "Average cost per share ($)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.2f",
        )
        save_position = st.form_submit_button("Save position", type="primary")

    if save_position:
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", manual_ticker):
            st.error("Enter a valid ticker, for example AAPL, BRK.B or NU.")
        else:
            position = {
                "Ticker": manual_ticker,
                "Shares": float(manual_shares),
                "Average Cost": float(manual_cost),
            }
            existing_index = next(
                (
                    index
                    for index, item in enumerate(st.session_state.manual_portfolio)
                    if item["Ticker"] == manual_ticker
                ),
                None,
            )
            if existing_index is None:
                st.session_state.manual_portfolio.append(position)
                st.success(f"{manual_ticker} was added to the portfolio.")
            else:
                st.session_state.manual_portfolio[existing_index] = position
                st.success(f"{manual_ticker} was updated.")

    restore_file = st.file_uploader(
        "Restore saved manual portfolio",
        type=["csv"],
        key="manual_portfolio_restore",
    )
    if restore_file is not None and st.button("Restore portfolio from CSV"):
        try:
            restored = pd.read_csv(restore_file)
            required = {"Ticker", "Shares", "Average Cost"}
            if not required.issubset(restored.columns):
                st.error("The file must contain Ticker, Shares and Average Cost columns.")
            else:
                clean_positions = []
                for _, row in restored.iterrows():
                    symbol = str(row["Ticker"]).strip().upper()
                    shares = float(row["Shares"])
                    average_cost = float(row["Average Cost"])
                    if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", symbol) and shares > 0 and average_cost >= 0:
                        clean_positions.append({
                            "Ticker": symbol,
                            "Shares": shares,
                            "Average Cost": average_cost,
                        })
                st.session_state.manual_portfolio = list({
                    item["Ticker"]: item for item in clean_positions
                }.values())
                st.success(f"Restored {len(clean_positions)} saved position(s).")
        except Exception as exc:
            st.error(f"The portfolio could not be restored: {exc}")

    if st.session_state.manual_portfolio:
        manual_df = pd.DataFrame(st.session_state.manual_portfolio)
        st.dataframe(
            manual_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Shares": st.column_config.NumberColumn(format="%.6f"),
                "Average Cost": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
        a1, a2, a3 = st.columns([1, 1, 2])
        remove_ticker = a1.selectbox(
            "Position to remove",
            manual_df["Ticker"].tolist(),
        )
        if a2.button("Remove position"):
            st.session_state.manual_portfolio = [
                item
                for item in st.session_state.manual_portfolio
                if item["Ticker"] != remove_ticker
            ]
            st.rerun()
        a3.download_button(
            "Download saved portfolio CSV",
            manual_df.to_csv(index=False),
            "stock_analyzer_portfolio.csv",
            "text/csv",
        )

        if st.button("Refresh portfolio analysis", type="primary"):
            portfolio_rows = []
            with st.spinner("Analyzing saved positions…"):
                for position in st.session_state.manual_portfolio:
                    symbol = position["Ticker"]
                    try:
                        analysis = analyze(symbol, "1y")
                        if not analysis:
                            continue
                        h, i, n, t, fs, fv, rs, values, news_score, news_risk, quality = analysis
                        shares = position["Shares"]
                        average_cost = position["Average Cost"]
                        current_value = shares * t["p"]
                        cost_basis = shares * average_cost
                        gain_loss = current_value - cost_basis if average_cost > 0 else np.nan
                        gain_loss_pct = (
                            gain_loss / cost_basis * 100
                            if average_cost > 0 and cost_basis
                            else np.nan
                        )
                        portfolio_rows.append({
                            "Ticker": symbol,
                            "Shares": shares,
                            "Average Cost": average_cost,
                            "Price": t["p"],
                            "Market Value": current_value,
                            "Gain/Loss": gain_loss,
                            "Gain/Loss %": gain_loss_pct,
                            "Entry Score": round(values[0]),
                            "Exit Score": round(values[1]),
                            "Risk": round(rs),
                            "Market Quality": round(quality["quality_score"]),
                            "Market Strength": round((t.get("market_strength") or {}).get("score", 50)),
                        })
                    except Exception:
                        continue
            if portfolio_rows:
                st.dataframe(
                    pd.DataFrame(portfolio_rows),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Shares": st.column_config.NumberColumn(format="%.6f"),
                        "Average Cost": st.column_config.NumberColumn(format="$%.2f"),
                        "Price": st.column_config.NumberColumn(format="$%.2f"),
                        "Market Value": st.column_config.NumberColumn(format="$%.2f"),
                        "Gain/Loss": st.column_config.NumberColumn(format="$%.2f"),
                        "Gain/Loss %": st.column_config.NumberColumn(format="%.2f%%"),
                    },
                )
            else:
                st.warning("No saved position could be analyzed right now.")
    else:
        st.info("No manual positions saved yet.")

    st.divider()
    st.subheader("Fidelity CSV Import")
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
                        h, i, n, t, fs, fv, rs, v, news_score, news_risk, quality = a
                        rows.append({"Ticker":sym,"Price":t["p"],"Entry Score":round(v[0]),"Exit Score":round(v[1]),"Risk":round(rs),"Fundamental":round(fs),"Market Strength":round((t.get("market_strength") or {}).get("score",50)),"Entry Low":v[2],"Entry High":v[3],"Stop":v[4],"Target 1":v[5],"R/R":round(v[7],2)})
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
st.sidebar.caption("V1.5 • Rule-based research prototype • No order execution")
