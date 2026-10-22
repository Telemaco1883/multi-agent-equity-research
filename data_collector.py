"""Market data and news collection for the equity research pipeline."""
from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

NEWS_URL = "https://news.google.com/rss/search?q={ticker}+stock"


def _num(value: Any) -> float | None:
    """Convert to float, returning None for missing/invalid values."""
    try:
        if value is None:
            return None
        f = float(value)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _rsi(close: pd.Series, period: int = 14) -> float | None:
    """Compute the Wilder RSI of a closing-price series."""
    if len(close) <= period:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    last_gain, last_loss = gain.iloc[-1], loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    return float(100 - 100 / (1 + last_gain / last_loss))


def get_stock_fundamentals(ticker: str) -> dict:
    """Pull valuation, margins, growth, balance sheet and technical data via yfinance.

    Returns a dict; on failure it contains an "error" key and no exception is raised.
    """
    ticker = ticker.strip().upper()
    data: dict[str, Any] = {"ticker": ticker}
    try:
        stock = yf.Ticker(ticker)
        try:
            info = stock.info or {}
        except Exception:
            info = {}

        hist = stock.history(period="1y")
        if hist is None or hist.empty:
            hist = yf.download(ticker, period="1y", progress=False, threads=False)
        if hist is None or hist.empty:
            hist = stock.history(period="6mo")
        if hist is None or hist.empty:
            hist = yf.download(ticker, period="6mo", progress=False, threads=False)
    except Exception as exc:
        return {"ticker": ticker, "error": f"Failed to fetch market data: {exc}"}

    if hist is None or hist.empty:
        return {"ticker": ticker, "error": f"No market data found for '{ticker}'."}

    close = hist["Close"].dropna()
    fast_info = getattr(stock, "fast_info", {}) or {}
    price = _num(close.iloc[-1]) if len(close) else _num(fast_info.get("lastPrice"))
    data.update(
        {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "price": price,
            "market_cap": _num(info.get("marketCap")) or _num(fast_info.get("marketCap")),
            "valuation": {
                "trailing_pe": _num(info.get("trailingPE")),
                "forward_pe": _num(info.get("forwardPE")),
                "ev_to_ebitda": _num(info.get("enterpriseToEbitda")),
                "price_to_sales": _num(info.get("priceToSalesTrailing12Months")),
                "peg_ratio": _num(info.get("pegRatio")),
            },
            "margins": {
                "gross": _num(info.get("grossMargins")),
                "operating": _num(info.get("operatingMargins")),
                "net": _num(info.get("profitMargins")),
            },
            "growth": {
                "revenue_growth": _num(info.get("revenueGrowth")),
                "earnings_growth": _num(info.get("earningsGrowth")),
            },
            "balance_sheet": {
                "total_cash": _num(info.get("totalCash")),
                "total_debt": _num(info.get("totalDebt")),
                "debt_to_equity": _num(info.get("debtToEquity")),
                "current_ratio": _num(info.get("currentRatio")),
                "free_cash_flow": _num(info.get("freeCashflow")),
            },
            "technicals": {
                "ma_50": _num(close.tail(50).mean()) if len(close) >= 50 else None,
                "ma_200": _num(close.tail(200).mean()) if len(close) >= 200 else None,
                "rsi_14": _rsi(close),
                "52w_high": _num(close.max()) if len(close) else None,
                "52w_low": _num(close.min()) if len(close) else None,
            },
        }
    )
    return data


def get_latest_news(ticker: str, limit: int = 8) -> list[dict]:
    """Fetch recent headlines from Google News RSS.

    Returns a list of {"title", "snippet", "published", "link"} dicts, or an empty
    list if the feed is empty or unavailable.
    """
    url = NEWS_URL.format(ticker=quote_plus(ticker.strip()))
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []
    items: list[dict] = []
    for entry in (feed.entries or [])[: max(limit, 0)]:
        summary_html = entry.get("summary", "") or ""
        snippet = BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)
        items.append(
            {
                "title": entry.get("title", "").strip(),
                "snippet": snippet,
                "published": entry.get("published", ""),
                "link": entry.get("link", ""),
            }
        )
    return items
