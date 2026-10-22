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


def _fmp_get_optional(path: str, params: dict[str, Any] | None = None) -> Any:
    try:
        return _fmp_get(path, params, base_url=FMP_STABLE_BASE_URL)
    except RuntimeError as exc:
        # Keep paid/entitlement endpoint failures non-fatal.
        if "HTTP 402" in str(exc):
            return None
        return None


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


def get_stock_fundamentals(ticker: str) -> dict:
    """Pull valuation, margins, growth, balance sheet and technical data via FMP stable APIs."""
    ticker = ticker.strip().upper()
    if not ticker:
        return {"ticker": ticker, "error": "Ticker is required."}

    try:
        profile = _first_record(
            _fmp_get("/profile", {"symbol": ticker}, base_url=FMP_STABLE_BASE_URL)
        )
    except Exception as exc:
        return {"ticker": ticker, "error": f"Failed to fetch market data: {exc}"}

    if not profile:
        return {"ticker": ticker, "error": f"No market data found for '{ticker}'."}

    income_payload = _fmp_get_optional(
        "/income-statement",
        {"symbol": ticker, "period": "annual", "limit": 2},
    )
    balance_payload = _fmp_get_optional(
        "/balance-sheet-statement",
        {"symbol": ticker, "period": "annual", "limit": 2},
    )
    cashflow_payload = _fmp_get_optional(
        "/cash-flow-statement",
        {"symbol": ticker, "period": "annual", "limit": 2},
    )
    ratios_payload = _fmp_get_optional("/ratios-ttm", {"symbol": ticker}) or _fmp_get_optional(
        "/ratios", {"symbol": ticker, "period": "annual", "limit": 1}
    )
    metrics_payload = _fmp_get_optional("/key-metrics-ttm", {"symbol": ticker}) or _fmp_get_optional(
        "/key-metrics", {"symbol": ticker, "period": "annual", "limit": 1}
    )

    income_latest, income_prior = _latest_statement(income_payload)
    balance_latest, _ = _latest_statement(balance_payload)
    cashflow_latest, _ = _latest_statement(cashflow_payload)

    ratios = _first_record(ratios_payload)
    metrics = _first_record(metrics_payload)

    revenue = _pick_number(income_latest.get("revenue"))
    prior_revenue = _pick_number(income_prior.get("revenue"))
    net_income = _pick_number(income_latest.get("netIncome"))
    prior_net_income = _pick_number(income_prior.get("netIncome"))

    revenue_growth = None
    if revenue is not None and prior_revenue not in (None, 0):
        revenue_growth = (revenue / prior_revenue) - 1.0
    earnings_growth = None
    if net_income is not None and prior_net_income not in (None, 0):
        earnings_growth = (net_income / prior_net_income) - 1.0

    gross_margin = _pick_number(ratios.get("grossProfitMarginTTM"), ratios.get("grossProfitMargin"))
    if gross_margin is None:
        gp = _pick_number(income_latest.get("grossProfit"))
        if gp is not None and revenue not in (None, 0):
            gross_margin = gp / revenue

    operating_margin = _pick_number(
        ratios.get("operatingProfitMarginTTM"), ratios.get("operatingProfitMargin")
    )
    if operating_margin is None:
        operating_income = _pick_number(income_latest.get("operatingIncome"))
        if operating_income is not None and revenue not in (None, 0):
            operating_margin = operating_income / revenue

    net_margin = _pick_number(ratios.get("netProfitMarginTTM"), ratios.get("netProfitMargin"))
    if net_margin is None and net_income is not None and revenue not in (None, 0):
        net_margin = net_income / revenue

    total_equity = _pick_number(
        balance_latest.get("totalStockholdersEquity"),
        balance_latest.get("totalEquity"),
    )
    total_debt = _pick_number(balance_latest.get("totalDebt"))
    debt_to_equity = _pick_number(
        ratios.get("debtToEquityRatioTTM"),
        ratios.get("debtEquityRatio"),
        ratios.get("debtToEquityRatio"),
    )
    if debt_to_equity is None and total_debt is not None and total_equity not in (None, 0):
        debt_to_equity = total_debt / total_equity

    current_ratio = _pick_number(ratios.get("currentRatioTTM"), ratios.get("currentRatio"))
    if current_ratio is None:
        current_assets = _pick_number(balance_latest.get("totalCurrentAssets"))
        current_liabilities = _pick_number(balance_latest.get("totalCurrentLiabilities"))
        if current_assets is not None and current_liabilities not in (None, 0):
            current_ratio = current_assets / current_liabilities

    free_cash_flow = _pick_number(cashflow_latest.get("freeCashFlow"))
    if free_cash_flow is None:
        op_cf = _pick_number(cashflow_latest.get("operatingCashFlow"))
        capex = _pick_number(cashflow_latest.get("capitalExpenditure"))
        if op_cf is not None:
            free_cash_flow = op_cf + capex if capex is not None else op_cf

    price = _pick_number(profile.get("price"))
    year_low, year_high = _parse_range_low_high(profile.get("range"))

    return {
        "ticker": ticker,
        "name": profile.get("companyName") or ticker,
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "price": price,
        "market_cap": _pick_number(profile.get("marketCap")),
        "valuation": {
            "trailing_pe": _pick_number(
                ratios.get("priceEarningsRatioTTM"),
                ratios.get("priceEarningsRatio"),
                profile.get("pe"),
            ),
            "forward_pe": _pick_number(ratios.get("forwardPE"), metrics.get("forwardPE")),
            "ev_to_ebitda": _pick_number(
                metrics.get("evToEBITDATTM"),
                metrics.get("enterpriseValueOverEBITDATTM"),
                metrics.get("evToEbitda"),
            ),
            "price_to_sales": _pick_number(
                ratios.get("priceToSalesRatioTTM"),
                ratios.get("priceToSalesRatio"),
                profile.get("priceToSalesRatio"),
            ),
            "peg_ratio": _pick_number(
                ratios.get("priceToEarningsGrowthRatioTTM"),
                ratios.get("pegRatio"),
            ),
        },
        "margins": {
            "gross": gross_margin,
            "operating": operating_margin,
            "net": net_margin,
        },
        "growth": {
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
        },
        "balance_sheet": {
            "total_cash": _pick_number(
                balance_latest.get("cashAndCashEquivalents"),
                balance_latest.get("cashAndShortTermInvestments"),
            ),
            "total_debt": total_debt,
            "debt_to_equity": debt_to_equity,
            "current_ratio": current_ratio,
            "free_cash_flow": free_cash_flow,
        },
        "technicals": {
            "ma_50": _pick_number(profile.get("priceAvg50")),
            "ma_200": _pick_number(profile.get("priceAvg200")),
            "rsi_14": None,
            "52w_high": year_high,
            "52w_low": year_low,
        },
    }


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
