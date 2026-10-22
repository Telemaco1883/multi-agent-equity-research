"""Market data and news collection for the equity research pipeline."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen
from typing import Any

import feedparser
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

NEWS_URL = "https://news.google.com/rss/search?q={ticker}+stock"
YAHOO_NEWS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
ARTICLE_TIMEOUT_SECONDS = 8
ARTICLE_MAX_CHARS = 3500


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


def _pick_number(*values: Any) -> float | None:
    for value in values:
        n = _num(value)
        if n is not None:
            return n
    return None


def _extract_close_series(hist: pd.DataFrame, ticker: str) -> pd.Series:
    if hist is None or hist.empty:
        return pd.Series(dtype="float64")
    if isinstance(hist, pd.Series):
        return hist.dropna()

    for key in ("Close", "close", "Adj Close", "adj close"):
        if key in hist.columns:
            return pd.to_numeric(hist[key], errors="coerce").dropna()

    if isinstance(hist.columns, pd.MultiIndex):
        for key in ("Close", "close", "Adj Close", "adj close"):
            cols = [c for c in hist.columns if str(c[-1]).upper() == ticker and str(c[0]) == key]
            if cols:
                return pd.to_numeric(hist[cols[0]], errors="coerce").dropna()
            cols = [c for c in hist.columns if str(c[0]) == key]
            if cols:
                series = hist[cols[0]]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                return pd.to_numeric(series, errors="coerce").dropna()
    return pd.Series(dtype="float64")


def _statement_value(frame: pd.DataFrame | None, candidates: list[str]) -> float | None:
    if frame is None or frame.empty:
        return None
    for label in candidates:
        if label in frame.index:
            row = pd.to_numeric(frame.loc[label], errors="coerce").dropna()
            if not row.empty:
                return _num(row.iloc[0])
    return None


def _statement_growth(frame: pd.DataFrame | None, row_label: str) -> float | None:
    if frame is None or frame.empty or row_label not in frame.index:
        return None
    row = pd.to_numeric(frame.loc[row_label], errors="coerce").dropna()
    if len(row) < 2:
        return None
    latest = _num(row.iloc[0])
    prev = _num(row.iloc[1])
    if latest is None or prev in (None, 0):
        return None
    return (latest - prev) / abs(prev)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _fetch_html(url: str, timeout: int = ARTICLE_TIMEOUT_SECONDS) -> str | None:
    if not url:
        return None
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="ignore")
    except (URLError, TimeoutError, ValueError):
        return None


def _extract_ldjson_text(soup: BeautifulSoup) -> str:
    best = ""
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        stack = parsed if isinstance(parsed, list) else [parsed]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key in ("articleBody", "description"):
                    value = node.get(key)
                    if isinstance(value, str):
                        cleaned = _clean_text(value)
                        if len(cleaned) > len(best):
                            best = cleaned
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(node, list):
                stack.extend(node)
    return best[:ARTICLE_MAX_CHARS]


def _sanitize_article_text(text: str, fallback: str) -> str:
    cleaned = _clean_text(text)[:ARTICLE_MAX_CHARS]
    if not cleaned:
        return fallback
    if "Comprehensive, up-to-date news coverage" in cleaned:
        return fallback
    return cleaned


def _extract_article_text(url: str, snippet: str) -> str:
    html = _fetch_html(url)
    if not html:
        return snippet

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    ldjson_text = _extract_ldjson_text(soup)
    if len(ldjson_text) >= 200:
        return _sanitize_article_text(ldjson_text, snippet)

    article = soup.find("article")
    if article:
        paragraphs = [_clean_text(p.get_text(" ", strip=True)) for p in article.find_all("p")]
        article_text = " ".join(p for p in paragraphs if len(p) > 40).strip()
        if len(article_text) >= 200:
            return _sanitize_article_text(article_text, snippet)

    meta_candidates = [
        soup.find("meta", attrs={"property": "og:description"}),
        soup.find("meta", attrs={"name": "description"}),
        soup.find("meta", attrs={"name": "twitter:description"}),
    ]
    for tag in meta_candidates:
        if tag and tag.get("content"):
            content = _clean_text(str(tag["content"]))
            if len(content) >= 80:
                return _sanitize_article_text(content, snippet)

    paragraphs = [_clean_text(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
    body_text = " ".join(p for p in paragraphs if len(p) > 40).strip()
    if len(body_text) >= 120:
        return _sanitize_article_text(body_text, snippet)
    return snippet


def _pick_source_url(entry: Any, summary_html: str) -> str:
    candidates: list[str] = []
    link = (entry.get("link", "") or "").strip()
    if link:
        candidates.append(link)
    for link_obj in entry.get("links", []) or []:
        href = (link_obj.get("href", "") or "").strip()
        if href:
            candidates.append(href)

    soup = BeautifulSoup(summary_html or "", "html.parser")
    for a in soup.find_all("a", href=True):
        href = (a["href"] or "").strip()
        if href:
            candidates.append(href)

    for candidate in candidates:
        parsed = urlparse(candidate)
        host = parsed.netloc.lower()
        if parsed.scheme not in {"http", "https"}:
            continue
        if "news.google.com" in host:
            continue
        return candidate
    return link


def _enrich_item(entry: Any) -> dict:
    summary_html = entry.get("summary", "") or ""
    snippet = BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)
    source_link = _pick_source_url(entry, summary_html)
    article_text = _extract_article_text(source_link, snippet)
    return {
        "title": entry.get("title", "").strip(),
        "snippet": snippet,
        "article_text": article_text,
        "published": entry.get("published", ""),
        "link": source_link or entry.get("link", ""),
    }


def _news_from_yfinance(ticker: str, limit: int) -> list[dict]:
    try:
        items = (yf.Ticker(ticker).news or [])[: max(limit, 0)]
    except Exception:
        return []

    base_items: list[dict] = []
    for item in items:
        title = (item.get("title", "") or "").strip()
        link = (item.get("link", "") or "").strip()
        if not title or not link:
            continue
        ts = item.get("providerPublishTime")
        published = ""
        if isinstance(ts, (int, float)):
            published = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        snippet = _clean_text((item.get("summary", "") or item.get("publisher", "") or "").strip())
        base_items.append(
            {
                "title": title,
                "snippet": snippet,
                "published": published,
                "link": link,
            }
        )

    if not base_items:
        return []

    def enrich(item: dict) -> dict:
        article_text = _extract_article_text(item["link"], item["snippet"])
        return {**item, "article_text": article_text}

    with ThreadPoolExecutor(max_workers=min(6, len(base_items))) as pool:
        return list(pool.map(enrich, base_items))


def _news_from_yahoo_rss(ticker: str, limit: int) -> list[dict]:
    url = YAHOO_NEWS_URL.format(ticker=quote_plus(ticker))
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []
    entries = (feed.entries or [])[: max(limit, 0)]
    if not entries:
        return []

    base_items: list[dict] = []
    for entry in entries:
        title = (entry.get("title", "") or "").strip()
        link = (entry.get("link", "") or "").strip()
        if not title or not link:
            continue
        summary_html = entry.get("summary", "") or ""
        snippet = BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)
        base_items.append(
            {
                "title": title,
                "snippet": snippet,
                "published": entry.get("published", ""),
                "link": link,
            }
        )

    if not base_items:
        return []

    def enrich(item: dict) -> dict:
        article_text = _extract_article_text(item["link"], item["snippet"])
        return {**item, "article_text": article_text}

    with ThreadPoolExecutor(max_workers=min(6, len(base_items))) as pool:
        return list(pool.map(enrich, base_items))


def get_stock_fundamentals(ticker: str) -> dict:
    """Pull valuation, margins, growth, balance sheet and technical data via yfinance."""
    ticker = ticker.strip().upper()
    if not ticker:
        return {"ticker": ticker, "error": "Ticker is required."}

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        hist = stock.history(period="1y", auto_adjust=False)
        income_stmt = stock.income_stmt
        balance_sheet_stmt = stock.balance_sheet
        cashflow_stmt = stock.cashflow
        fast_info_raw = getattr(stock, "fast_info", None)
    except Exception as exc:
        return {"ticker": ticker, "error": f"Failed to fetch market data: {exc}"}

    close = _extract_close_series(hist, ticker)
    if close.empty:
        return {"ticker": ticker, "error": f"No market data found for '{ticker}'."}

    fast_info: dict[str, Any] = {}
    if isinstance(fast_info_raw, dict):
        fast_info = fast_info_raw
    elif fast_info_raw is not None:
        try:
            fast_info = dict(fast_info_raw)
        except (TypeError, ValueError):
            fast_info = {}

    price = _pick_number(
        close.iloc[-1] if len(close) else None,
        fast_info.get("lastPrice"),
        fast_info.get("last_price"),
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
        info.get("previousClose"),
    )
    market_cap = _pick_number(
        fast_info.get("marketCap"),
        fast_info.get("market_cap"),
        info.get("marketCap"),
    )
    revenue_stmt = _statement_value(income_stmt, ["Total Revenue", "Operating Revenue"])
    net_income_stmt = _statement_value(
        income_stmt,
        ["Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"],
    )
    gross_profit_stmt = _statement_value(income_stmt, ["Gross Profit"])
    operating_income_stmt = _statement_value(
        income_stmt,
        ["Operating Income", "Operating Income Loss", "EBIT"],
    )
    current_assets_stmt = _statement_value(
        balance_sheet_stmt,
        ["Current Assets", "Total Current Assets"],
    )
    current_liabilities_stmt = _statement_value(
        balance_sheet_stmt,
        ["Current Liabilities", "Total Current Liabilities"],
    )
    total_equity_stmt = _statement_value(
        balance_sheet_stmt,
        ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"],
    )
    total_cash_stmt = _statement_value(
        balance_sheet_stmt,
        [
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Short Term Investments",
        ],
    )
    total_debt_stmt = _statement_value(
        balance_sheet_stmt,
        ["Total Debt", "Current Debt And Capital Lease Obligation"],
    )
    free_cash_flow_stmt = _statement_value(cashflow_stmt, ["Free Cash Flow"])
    revenue_growth_stmt = _statement_growth(income_stmt, "Total Revenue")
    earnings_growth_stmt = _statement_growth(income_stmt, "Net Income")
    ev_to_ebitda_info = _pick_number(info.get("enterpriseToEbitda"))
    enterprise_value = _pick_number(info.get("enterpriseValue"))
    ebitda = _pick_number(info.get("ebitda"))
    ev_to_ebitda_derived = (
        enterprise_value / ebitda
        if enterprise_value is not None and ebitda not in (None, 0)
        else None
    )

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "price": price,
        "market_cap": market_cap,
        "valuation": {
            "trailing_pe": _pick_number(info.get("trailingPE")),
            "forward_pe": _pick_number(info.get("forwardPE")),
            "ev_to_ebitda": _pick_number(ev_to_ebitda_info, ev_to_ebitda_derived),
            "price_to_sales": _pick_number(info.get("priceToSalesTrailing12Months")),
            "peg_ratio": _pick_number(info.get("pegRatio")),
        },
        "margins": {
            "gross": _pick_number(
                info.get("grossMargins"),
                (gross_profit_stmt / revenue_stmt)
                if gross_profit_stmt is not None and revenue_stmt not in (None, 0)
                else None,
            ),
            "operating": _pick_number(
                info.get("operatingMargins"),
                (operating_income_stmt / revenue_stmt)
                if operating_income_stmt is not None and revenue_stmt not in (None, 0)
                else None,
            ),
            "net": _pick_number(
                info.get("profitMargins"),
                (net_income_stmt / revenue_stmt)
                if net_income_stmt is not None and revenue_stmt not in (None, 0)
                else None,
            ),
        },
        "growth": {
            "revenue_growth": _pick_number(info.get("revenueGrowth"), revenue_growth_stmt),
            "earnings_growth": _pick_number(info.get("earningsGrowth"), earnings_growth_stmt),
        },
        "balance_sheet": {
            "total_cash": _pick_number(info.get("totalCash"), total_cash_stmt),
            "total_debt": _pick_number(info.get("totalDebt"), total_debt_stmt),
            "debt_to_equity": _pick_number(
                info.get("debtToEquity"),
                ((total_debt_stmt / total_equity_stmt) * 100)
                if total_debt_stmt is not None and total_equity_stmt not in (None, 0)
                else None,
            ),
            "current_ratio": _pick_number(
                info.get("currentRatio"),
                (current_assets_stmt / current_liabilities_stmt)
                if current_assets_stmt is not None and current_liabilities_stmt not in (None, 0)
                else None,
            ),
            "free_cash_flow": _pick_number(info.get("freeCashflow"), free_cash_flow_stmt),
        },
        "technicals": {
            "ma_50": _num(close.tail(50).mean()) if len(close) >= 50 else None,
            "ma_200": _num(close.tail(200).mean()) if len(close) >= 200 else None,
            "rsi_14": _rsi(close),
            "52w_high": _pick_number(
                _num(close.max()) if len(close) else None,
                fast_info.get("yearHigh"),
                info.get("fiftyTwoWeekHigh"),
            ),
            "52w_low": _pick_number(
                _num(close.min()) if len(close) else None,
                fast_info.get("yearLow"),
                info.get("fiftyTwoWeekLow"),
            ),
        },
    }


def get_latest_news(ticker: str, limit: int = 8) -> list[dict]:
    """Fetch recent news from Google News RSS and extract article text when possible.

    Returns a list of {"title", "snippet", "article_text", "published", "link"} dicts,
    or an empty list if the feed is empty or unavailable.
    """
    ticker = ticker.strip().upper()
    max_items = max(limit, 0)
    if max_items == 0:
        return []

    news = _news_from_yfinance(ticker, max_items)
    if len(news) < max_items:
        yahoo_rss_news = _news_from_yahoo_rss(ticker, max_items)
        seen_yahoo = {n.get("link", "") for n in news}
        for item in yahoo_rss_news:
            link = item.get("link", "")
            if link and link in seen_yahoo:
                continue
            news.append(item)
            if link:
                seen_yahoo.add(link)
            if len(news) >= max_items:
                break
    if len(news) >= max_items:
        return news[:max_items]

    url = NEWS_URL.format(ticker=quote_plus(ticker))
    try:
        feed = feedparser.parse(url)
    except Exception:
        return news[:max_items]
    entries = (feed.entries or [])[:max_items]
    if not entries:
        return news[:max_items]

    with ThreadPoolExecutor(max_workers=min(6, len(entries))) as pool:
        google_news = list(pool.map(_enrich_item, entries))

    seen = {n.get("link", "") for n in news}
    for item in google_news:
        link = item.get("link", "")
        if link and link in seen:
            continue
        news.append(item)
        if link:
            seen.add(link)
        if len(news) >= max_items:
            break
    return news[:max_items]
