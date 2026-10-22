"""Market data and news collection for the equity research pipeline."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any
from urllib.error import URLError
from urllib.parse import quote_plus, urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup

import config

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
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


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _pct_growth(latest: float | None, previous: float | None) -> float | None:
    if latest is None or previous in (None, 0):
        return None
    return (latest - previous) / abs(previous)


def _fmp_get(path: str, params: dict[str, Any] | None = None) -> Any:
    if not config.FMP_API_KEY:
        raise RuntimeError("Missing FMP_API_KEY. Set it in your environment or .env file.")
    query = {"apikey": config.FMP_API_KEY}
    if params:
        query.update(params)
    url = f"{FMP_BASE_URL}{path}?{urlencode(query, doseq=True)}"
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
        with urlopen(request, timeout=12) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            payload = response.read().decode(charset, errors="ignore")
            return json.loads(payload)
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"FMP request failed for {path}: {exc}") from exc


def _first_record(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    if isinstance(payload, dict):
        return payload
    return {}


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


def get_stock_fundamentals(ticker: str) -> dict:
    """Pull valuation, margins, growth, balance sheet and technical data via FMP API."""
    ticker = ticker.strip().upper()
    if not ticker:
        return {"ticker": ticker, "error": "Ticker is required."}

    try:
        quote = _first_record(_fmp_get(f"/quote/{quote_plus(ticker)}"))
        profile = _first_record(_fmp_get(f"/profile/{quote_plus(ticker)}"))
        metrics_ttm = _first_record(_fmp_get(f"/key-metrics-ttm/{quote_plus(ticker)}"))
        ratios_ttm = _first_record(_fmp_get(f"/ratios-ttm/{quote_plus(ticker)}"))
        income = _fmp_get(f"/income-statement/{quote_plus(ticker)}", {"limit": 2})
        balance = _fmp_get(f"/balance-sheet-statement/{quote_plus(ticker)}", {"limit": 1})
        cashflow = _fmp_get(f"/cash-flow-statement/{quote_plus(ticker)}", {"limit": 1})
        price_history = _fmp_get(f"/historical-price-full/{quote_plus(ticker)}", {"serietype": "line"})
    except Exception as exc:
        return {"ticker": ticker, "error": f"Failed to fetch market data: {exc}"}

    historical = price_history.get("historical", []) if isinstance(price_history, dict) else []
    close = pd.to_numeric(
        pd.Series([entry.get("close") for entry in historical if isinstance(entry, dict)]),
        errors="coerce",
    ).dropna()
    if close.empty:
        return {"ticker": ticker, "error": f"No market data found for '{ticker}'."}

    income_rows = [row for row in income if isinstance(row, dict)] if isinstance(income, list) else []
    latest_income = income_rows[0] if income_rows else {}
    previous_income = income_rows[1] if len(income_rows) > 1 else {}
    latest_balance = _first_record(balance)
    latest_cashflow = _first_record(cashflow)

    revenue = _pick_number(latest_income.get("revenue"))
    revenue_prev = _pick_number(previous_income.get("revenue"))
    net_income = _pick_number(latest_income.get("netIncome"))
    net_income_prev = _pick_number(previous_income.get("netIncome"))
    gross_profit = _pick_number(latest_income.get("grossProfit"))
    operating_income = _pick_number(latest_income.get("operatingIncome"))
    total_cash = _pick_number(
        latest_balance.get("cashAndCashEquivalents"),
        latest_balance.get("cashAndShortTermInvestments"),
    )
    total_debt = _pick_number(latest_balance.get("totalDebt"))
    total_equity = _pick_number(latest_balance.get("totalStockholdersEquity"))
    current_assets = _pick_number(latest_balance.get("totalCurrentAssets"))
    current_liabilities = _pick_number(latest_balance.get("totalCurrentLiabilities"))
    free_cash_flow = _pick_number(latest_cashflow.get("freeCashFlow"))
    debt_to_equity_ratio = _safe_div(total_debt, total_equity)

    return {
        "ticker": ticker,
        "name": profile.get("companyName") or quote.get("name") or ticker,
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "price": _pick_number(quote.get("price"), close.iloc[-1] if len(close) else None),
        "market_cap": _pick_number(quote.get("marketCap")),
        "valuation": {
            "trailing_pe": _pick_number(quote.get("pe"), ratios_ttm.get("priceToEarningsRatioTTM")),
            "forward_pe": _pick_number(quote.get("pe")),  # FMP free endpoints usually expose trailing only.
            "ev_to_ebitda": _pick_number(metrics_ttm.get("enterpriseValueOverEBITDATTM")),
            "price_to_sales": _pick_number(
                quote.get("priceToSalesRatio"),
                ratios_ttm.get("priceToSalesRatioTTM"),
            ),
            "peg_ratio": _pick_number(ratios_ttm.get("priceEarningsToGrowthRatioTTM")),
        },
        "margins": {
            "gross": _pick_number(
                ratios_ttm.get("grossProfitMarginTTM"),
                _safe_div(gross_profit, revenue),
            ),
            "operating": _pick_number(
                ratios_ttm.get("operatingProfitMarginTTM"),
                _safe_div(operating_income, revenue),
            ),
            "net": _pick_number(
                ratios_ttm.get("netProfitMarginTTM"),
                _safe_div(net_income, revenue),
            ),
        },
        "growth": {
            "revenue_growth": _pick_number(_pct_growth(revenue, revenue_prev)),
            "earnings_growth": _pick_number(_pct_growth(net_income, net_income_prev)),
        },
        "balance_sheet": {
            "total_cash": total_cash,
            "total_debt": total_debt,
            "debt_to_equity": _pick_number(
                ratios_ttm.get("debtEquityRatioTTM"),
                debt_to_equity_ratio * 100 if debt_to_equity_ratio is not None else None,
            ),
            "current_ratio": _pick_number(
                ratios_ttm.get("currentRatioTTM"),
                _safe_div(current_assets, current_liabilities),
            ),
            "free_cash_flow": free_cash_flow,
        },
        "technicals": {
            "ma_50": _num(close.tail(50).mean()) if len(close) >= 50 else None,
            "ma_200": _num(close.tail(200).mean()) if len(close) >= 200 else None,
            "rsi_14": _rsi(close),
            "52w_high": _pick_number(quote.get("yearHigh"), _num(close.max())),
            "52w_low": _pick_number(quote.get("yearLow"), _num(close.min())),
        },
    }


def get_latest_news(ticker: str, limit: int = 8) -> list[dict]:
    """Fetch recent company news from FMP and enrich article text when possible."""
    ticker = ticker.strip().upper()
    max_items = max(limit, 0)
    if max_items == 0:
        return []
    if not config.FMP_API_KEY:
        return []

    try:
        payload = _fmp_get("/stock_news", {"tickers": ticker, "limit": max_items})
    except Exception:
        return []
    items = [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []
    if not items:
        return []

    base_items: list[dict[str, str]] = []
    for item in items[:max_items]:
        title = _clean_text(str(item.get("title", "")))
        link = _clean_text(str(item.get("url", "")))
        snippet = _clean_text(str(item.get("text", "")))
        if not title or not link:
            continue
        published_raw = _clean_text(str(item.get("publishedDate", "")))
        published = published_raw
        if published_raw:
            try:
                published = datetime.fromisoformat(published_raw.replace("Z", "+00:00")).astimezone(
                    timezone.utc
                ).isoformat()
            except ValueError:
                pass
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

    def enrich(item: dict[str, str]) -> dict[str, str]:
        parsed = urlparse(item["link"])
        if parsed.scheme not in {"http", "https"}:
            return {**item, "article_text": item["snippet"]}
        article_text = _extract_article_text(item["link"], item["snippet"])
        return {**item, "article_text": article_text}

    with ThreadPoolExecutor(max_workers=min(6, len(base_items))) as pool:
        return list(pool.map(enrich, base_items))
