"""Market data and news collection for the equity research pipeline."""
from __future__ import annotations

import json
import re
import time
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
YF_QUOTE_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules={modules}"
YF_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range={range_}&interval=1d"
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


def _obj_get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    try:
        if isinstance(obj, dict):
            return obj.get(key)
        getter = getattr(obj, "get", None)
        if callable(getter):
            return getter(key)
        return getattr(obj, key, None)
    except Exception:
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


def _yf_raw(value: Any) -> Any:
    if isinstance(value, dict):
        if "raw" in value:
            return value.get("raw")
        if "fmt" in value and value.get("fmt") is not None:
            return value.get("fmt")
    return value


def _nested_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _yf_raw(current)


def _fetch_json(url: str, timeout: int = 8, retries: int = 3, backoff: float = 1.3) -> dict[str, Any] | None:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
    )
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                payload = response.read().decode(charset, errors="ignore")
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    return parsed
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError):
            pass
        if attempt < retries - 1:
            time.sleep(backoff**attempt)
    return None


def _fetch_quote_summary(ticker: str) -> dict[str, Any]:
    modules = ",".join(
        ["price", "summaryDetail", "financialData", "defaultKeyStatistics", "assetProfile"]
    )
    url = YF_QUOTE_SUMMARY_URL.format(ticker=quote_plus(ticker), modules=modules)
    payload = _fetch_json(url)
    if not payload:
        return {}
    result = payload.get("quoteSummary", {}).get("result") or []
    if result and isinstance(result[0], dict):
        return result[0]
    return {}


def _fetch_chart_close(ticker: str, range_: str = "1y") -> pd.Series:
    url = YF_CHART_URL.format(ticker=quote_plus(ticker), range_=range_)
    payload = _fetch_json(url)
    if not payload:
        return pd.Series(dtype="float64")
    result = payload.get("chart", {}).get("result") or []
    if not result:
        return pd.Series(dtype="float64")
    indicators = result[0].get("indicators", {})
    quote = (indicators.get("quote") or [{}])[0]
    close = quote.get("close") or []
    return pd.to_numeric(pd.Series(close), errors="coerce").dropna()


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
        quote_summary = _fetch_quote_summary(ticker)
    except Exception as exc:
        return {"ticker": ticker, "error": f"Failed to fetch market data: {exc}"}

    if hist is None or hist.empty:
        chart_close = _fetch_chart_close(ticker, "1y")
        if chart_close.empty:
            chart_close = _fetch_chart_close(ticker, "6mo")
        if chart_close.empty:
            return {"ticker": ticker, "error": f"No market data found for '{ticker}'."}
        close = chart_close
    else:
        close = _extract_close_series(hist, ticker)

    if close.empty:
        return {"ticker": ticker, "error": f"No market data found for '{ticker}'."}
    fast_info = getattr(stock, "fast_info", None)
    summary_price = quote_summary.get("price", {})
    summary_detail = quote_summary.get("summaryDetail", {})
    financial_data = quote_summary.get("financialData", {})
    default_stats = quote_summary.get("defaultKeyStatistics", {})
    asset_profile = quote_summary.get("assetProfile", {})

    price = _pick_number(
        close.iloc[-1] if len(close) else None,
        _obj_get(fast_info, "lastPrice"),
        _obj_get(fast_info, "last_price"),
        _nested_get(summary_price, "regularMarketPrice"),
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
        info.get("previousClose"),
    )
    market_cap = _pick_number(
        info.get("marketCap"),
        _obj_get(fast_info, "marketCap"),
        _obj_get(fast_info, "market_cap"),
        _nested_get(summary_price, "marketCap"),
        _nested_get(default_stats, "marketCap"),
    )
    data.update(
        {
            "name": (
                info.get("longName")
                or info.get("shortName")
                or _nested_get(summary_price, "longName")
                or _nested_get(summary_price, "shortName")
                or ticker
            ),
            "sector": info.get("sector") or _nested_get(asset_profile, "sector"),
            "industry": info.get("industry") or _nested_get(asset_profile, "industry"),
            "price": price,
            "market_cap": market_cap,
            "valuation": {
                "trailing_pe": _pick_number(
                    info.get("trailingPE"),
                    _nested_get(summary_detail, "trailingPE"),
                    _nested_get(default_stats, "trailingPE"),
                ),
                "forward_pe": _pick_number(
                    info.get("forwardPE"),
                    _nested_get(default_stats, "forwardPE"),
                    _nested_get(summary_detail, "forwardPE"),
                ),
                "ev_to_ebitda": _pick_number(
                    info.get("enterpriseToEbitda"),
                    _nested_get(default_stats, "enterpriseToEbitda"),
                ),
                "price_to_sales": _pick_number(
                    info.get("priceToSalesTrailing12Months"),
                    _nested_get(default_stats, "priceToSalesTrailing12Months"),
                ),
                "peg_ratio": _pick_number(
                    info.get("pegRatio"),
                    _nested_get(default_stats, "pegRatio"),
                ),
            },
            "margins": {
                "gross": _pick_number(
                    info.get("grossMargins"),
                    _nested_get(financial_data, "grossMargins"),
                ),
                "operating": _pick_number(
                    info.get("operatingMargins"),
                    _nested_get(financial_data, "operatingMargins"),
                ),
                "net": _pick_number(
                    info.get("profitMargins"),
                    _nested_get(financial_data, "profitMargins"),
                ),
            },
            "growth": {
                "revenue_growth": _pick_number(
                    info.get("revenueGrowth"),
                    _nested_get(financial_data, "revenueGrowth"),
                ),
                "earnings_growth": _pick_number(
                    info.get("earningsGrowth"),
                    _nested_get(financial_data, "earningsGrowth"),
                ),
            },
            "balance_sheet": {
                "total_cash": _pick_number(
                    info.get("totalCash"),
                    _nested_get(financial_data, "totalCash"),
                ),
                "total_debt": _pick_number(
                    info.get("totalDebt"),
                    _nested_get(financial_data, "totalDebt"),
                ),
                "debt_to_equity": _pick_number(
                    info.get("debtToEquity"),
                    _nested_get(financial_data, "debtToEquity"),
                ),
                "current_ratio": _pick_number(
                    info.get("currentRatio"),
                    _nested_get(financial_data, "currentRatio"),
                ),
                "free_cash_flow": _pick_number(
                    info.get("freeCashflow"),
                    _nested_get(financial_data, "freeCashflow"),
                ),
            },
            "technicals": {
                "ma_50": _num(close.tail(50).mean()) if len(close) >= 50 else None,
                "ma_200": _num(close.tail(200).mean()) if len(close) >= 200 else None,
                "rsi_14": _rsi(close),
                "52w_high": _pick_number(
                    _num(close.max()) if len(close) else None,
                    _obj_get(fast_info, "yearHigh"),
                    _nested_get(summary_detail, "fiftyTwoWeekHigh"),
                ),
                "52w_low": _pick_number(
                    _num(close.min()) if len(close) else None,
                    _obj_get(fast_info, "yearLow"),
                    _nested_get(summary_detail, "fiftyTwoWeekLow"),
                ),
            },
        }
    )
    return data


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
