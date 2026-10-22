"""Market data and news collection for the equity research pipeline."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode, urlparse
from urllib.request import Request, urlopen

import feedparser
import pandas as pd
from bs4 import BeautifulSoup

import config

FMP_STABLE_BASE_URL = "https://financialmodelingprep.com/stable"
NEWS_URL = "https://news.google.com/rss/search?q={ticker}+stock"
STREET_COMMENTARY_QUERIES = [
    (
        "(\"macro outlook\" OR \"asset allocation\" OR \"strategy note\" OR "
        "\"chief investment officer\" OR \"market outlook\" OR \"global strategist\") AND "
        "(\"equities\" OR \"fixed income\" OR \"credit\" OR \"bonds\") AND "
        "(\"overweight\" OR \"underweight\" OR \"rate cut\" OR \"yield curve\") "
        "-earnings -quarterly when:7d"
    ),
    (
        "site:reuters.com OR site:marketwatch.com OR site:cnbc.com OR site:finance.yahoo.com "
        "AND (\"strategists say\" OR \"Wall Street consensus\" OR \"base case\" OR "
        "\"soft landing\" OR \"terminal rate\" OR \"rate cut forecast\" OR \"recession risk\") "
        "-earnings when:7d"
    ),
    (
        "(\"consensus forecast\" OR \"survey of forecasters\" OR \"median estimate\" OR "
        "\"economists survey\" OR \"fund manager survey\") AND "
        "(\"Fed\" OR \"inflation\" OR \"CPI\" OR \"Treasury yields\" OR \"GDP growth\") when:14d"
    ),
]
STREET_COMMENTARY_URL = "https://news.google.com/rss/search?q={query}"
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


def _pick_number(*values: Any) -> float | None:
    for value in values:
        n = _num(value)
        if n is not None:
            return n
    return None


def _parse_range_low_high(range_text: Any) -> tuple[float | None, float | None]:
    raw = str(range_text or "").strip()
    if not raw:
        return None, None
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", raw)
    if not match:
        return None, None
    low = _num(match.group(1))
    high = _num(match.group(2))
    return low, high


def _fmp_get(
    path: str,
    params: dict[str, Any] | None = None,
    *,
    base_url: str = FMP_STABLE_BASE_URL,
) -> Any:
    if not config.FMP_API_KEY:
        raise RuntimeError("Missing FMP_API_KEY. Set it in your environment or .env file.")
    query = {"apikey": config.FMP_API_KEY}
    if params:
        query.update(params)
    url = f"{base_url}{path}?{urlencode(query, doseq=True)}"
    request = Request(
        url,
        headers={
            "apikey": config.FMP_API_KEY,
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
    )
    try:
        with urlopen(request, timeout=12) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            payload = response.read().decode(charset, errors="ignore")
            return json.loads(payload)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                "FMP request was rejected (401/403). Check FMP_API_KEY, plan access, and key validity."
            ) from exc
        if exc.code == 402:
            raise RuntimeError(f"FMP request requires a paid plan for {path} (HTTP 402).") from exc
        raise RuntimeError(f"FMP request failed for {path}: {exc}") from exc
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"FMP request failed for {path}: {exc}") from exc


def _first_record(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    if isinstance(payload, dict):
        return payload
    return {}


def _latest_statement(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, list):
        return {}, {}
    rows = [row for row in payload if isinstance(row, dict)]
    if not rows:
        return {}, {}
    latest = rows[0]
    prior = rows[1] if len(rows) > 1 else {}
    return latest, prior


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
        if parsed.scheme not in {"http", "https"}:
            continue
        if "news.google.com" in parsed.netloc.lower():
            continue
        return candidate
    return link


def _enrich_google_entry(entry: Any) -> dict[str, str]:
    summary_html = entry.get("summary", "") or ""
    snippet = BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)
    source_link = _pick_source_url(entry, summary_html)
    return {
        "title": (entry.get("title", "") or "").strip(),
        "snippet": snippet,
        "article_text": _extract_article_text(source_link, snippet),
        "published": (entry.get("published", "") or "").strip(),
        "link": source_link or (entry.get("link", "") or "").strip(),
    }


def get_stock_fundamentals_raw(ticker: str) -> dict:
    """Fetch full, unmodified fundamentals payloads from FMP stable endpoints."""
    symbol = ticker.strip().upper()
    if not symbol:
        return {"ticker": symbol, "error": "Ticker is required."}

    try:
        return {
            "ticker": symbol,
            "profile": _fmp_get("/profile", {"symbol": symbol}, base_url=FMP_STABLE_BASE_URL),
            "income_statement": _fmp_get(
                "/income-statement",
                {"symbol": symbol, "period": "annual", "limit": 2},
                base_url=FMP_STABLE_BASE_URL,
            ),
            "balance_sheet_statement": _fmp_get(
                "/balance-sheet-statement",
                {"symbol": symbol, "period": "annual", "limit": 2},
                base_url=FMP_STABLE_BASE_URL,
            ),
            "cash_flow_statement": _fmp_get(
                "/cash-flow-statement",
                {"symbol": symbol, "period": "annual", "limit": 2},
                base_url=FMP_STABLE_BASE_URL,
            ),
            "ratios_ttm": _fmp_get(
                "/ratios-ttm",
                {"symbol": symbol},
                base_url=FMP_STABLE_BASE_URL,
            ),
            "key_metrics_ttm": _fmp_get(
                "/key-metrics-ttm",
                {"symbol": symbol},
                base_url=FMP_STABLE_BASE_URL,
            ),
        }
    except Exception as exc:
        return {"ticker": symbol, "error": f"Failed to fetch fundamentals: {exc}"}


def get_stock_fundamentals(ticker: str) -> dict:
    return get_stock_fundamentals_raw(ticker)


def get_latest_news(ticker: str, limit: int = 8) -> list[dict]:
    """Fetch recent company news from Google News RSS and enrich article text when possible."""
    ticker = ticker.strip().upper()
    max_items = max(limit, 0)
    if max_items == 0:
        return []

    try:
        feed = feedparser.parse(NEWS_URL.format(ticker=quote_plus(ticker)))
    except Exception:
        return []
    entries = (feed.entries or [])[:max_items]
    if not entries:
        return []

    with ThreadPoolExecutor(max_workers=min(6, len(entries))) as pool:
        enriched = list(pool.map(_enrich_google_entry, entries))
    return [item for item in enriched if item.get("title")]


def get_street_commentary(limit: int = 60) -> list[dict]:
    """Fetch recent macro commentary items from major street institutions."""
    max_items = max(limit, 0)
    if max_items == 0:
        return []

    entries: list[Any] = []
    per_query_limit = 20
    for query in STREET_COMMENTARY_QUERIES:
        try:
            feed = feedparser.parse(STREET_COMMENTARY_URL.format(query=quote_plus(query)))
        except Exception:
            continue
        entries.extend((feed.entries or [])[:per_query_limit])
    if not entries:
        return []

    with ThreadPoolExecutor(max_workers=min(6, len(entries))) as pool:
        enriched = list(pool.map(_enrich_google_entry, entries))

    seen: set[str] = set()
    unique: list[dict] = []
    for item in enriched:
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        content = (item.get("article_text") or item.get("snippet") or "").strip()
        key = link or title.lower()
        if not title or not key or key in seen or len(content) < 40:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= max_items:
            break
    return unique
