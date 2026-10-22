"""Market data and news collection for the equity research pipeline."""
from __future__ import annotations

import json
import math
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


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
    return best


def _sanitize_article_text(text: str, fallback: str) -> str:
    cleaned = _clean_text(text)
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


def _build_technical_payload(historical_payload: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if isinstance(historical_payload, list):
        rows = [row for row in historical_payload if isinstance(row, dict)]
    elif isinstance(historical_payload, dict):
        for key in ("historical", "prices", "data"):
            candidate = historical_payload.get(key)
            if isinstance(candidate, list):
                rows = [row for row in candidate if isinstance(row, dict)]
                if rows:
                    break
        if not rows:
            stock_list = historical_payload.get("historicalStockList")
            if isinstance(stock_list, list):
                for item in stock_list:
                    if not isinstance(item, dict):
                        continue
                    candidate = item.get("historical")
                    if isinstance(candidate, list):
                        rows = [row for row in candidate if isinstance(row, dict)]
                        if rows:
                            break
    if not rows:
        return {"technical_summary": {"error": "No historical price payload returned"}, "technical_chart": []}

    df = pd.DataFrame(rows)
    df.columns = [str(col).strip().lower() for col in df.columns]
    if "close" not in df.columns and "adjclose" in df.columns:
        df["close"] = df["adjclose"]
    required = {"date", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return {
            "technical_summary": {"error": f"Missing technical columns: {sorted(required - set(df.columns))}"},
            "technical_chart": [],
        }

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "high", "low", "close", "volume"]).sort_values("date")
    if len(df) < 200:
        return {
            "technical_summary": {"error": f"Insufficient price history ({len(df)} rows)"},
            "technical_chart": [],
        }

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    returns = close.pct_change()

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_100 = close.rolling(100).mean()
    sma_200 = close.rolling(200).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    rsi_14 = rsi_series.iloc[-1]

    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    std_20 = close.rolling(20).std()
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    bb_width = (bb_upper - bb_lower) / sma_20
    bb_position = (close - bb_lower) / (bb_upper - bb_lower)

    lowest_14 = low.rolling(14).min()
    highest_14 = high.rolling(14).max()
    stoch_k = ((close - lowest_14) / (highest_14 - lowest_14)) * 100
    stoch_d = stoch_k.rolling(3).mean()

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr_14 = true_range.rolling(14).mean()

    vol_20_ma = volume.rolling(20).mean().iloc[-1]
    current_vol = float(volume.iloc[-1])
    curr_price = float(close.iloc[-1])
    sma_200_last = float(sma_200.iloc[-1])
    price_vs_sma200 = ((curr_price - sma_200_last) / sma_200_last) * 100 if sma_200_last else None
    volume_vs_20d = ((current_vol - vol_20_ma) / vol_20_ma) * 100 if vol_20_ma else None
    volatility_20d = returns.rolling(20).std().iloc[-1] * (252**0.5) * 100

    momentum_21d = ((close.iloc[-1] / close.iloc[-22]) - 1) * 100 if len(close) > 22 else None
    momentum_63d = ((close.iloc[-1] / close.iloc[-64]) - 1) * 100 if len(close) > 64 else None
    momentum_126d = ((close.iloc[-1] / close.iloc[-127]) - 1) * 100 if len(close) > 127 else None
    momentum_252d = ((close.iloc[-1] / close.iloc[-253]) - 1) * 100 if len(close) > 253 else None

    chart_df = pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m-%d"),
            "close": close.round(2),
            "sma_20": sma_20.round(2),
            "sma_50": sma_50.round(2),
            "sma_100": sma_100.round(2),
            "sma_200": sma_200.round(2),
            "ema_21": ema_21.round(2),
            "ema_50": ema_50.round(2),
            "rsi_14": rsi_series.round(2),
            "macd": macd_line.round(4),
            "macd_signal": macd_signal.round(4),
            "macd_hist": macd_hist.round(4),
        }
    ).dropna(subset=["sma_200"])

    return _json_safe({
        "technical_summary": {
            "rows_used": int(len(df)),
            "start_date": str(df["date"].iloc[0].date()),
            "end_date": str(df["date"].iloc[-1].date()),
            "current_price": round(curr_price, 2),
            "52w_high": round(float(high.tail(252).max()), 2),
            "52w_low": round(float(low.tail(252).min()), 2),
            "sma_20": round(float(sma_20.iloc[-1]), 2),
            "sma_50": round(float(sma_50.iloc[-1]), 2),
            "sma_100": round(float(sma_100.iloc[-1]), 2),
            "sma_200": round(sma_200_last, 2),
            "ema_21": round(float(ema_21.iloc[-1]), 2),
            "ema_50": round(float(ema_50.iloc[-1]), 2),
            "price_vs_sma200_pct": round(price_vs_sma200, 2) if price_vs_sma200 is not None else None,
            "price_vs_sma50_pct": round(((curr_price - float(sma_50.iloc[-1])) / float(sma_50.iloc[-1])) * 100, 2),
            "ma_alignment": (
                "Bullish (SMA50 > SMA200)"
                if float(sma_50.iloc[-1]) > sma_200_last
                else "Bearish (SMA50 < SMA200)"
            ),
            "rsi_14": round(float(rsi_14), 2),
            "rsi_condition": (
                "Overbought (>70)"
                if float(rsi_14) > 70
                else ("Oversold (<30)" if float(rsi_14) < 30 else "Neutral (30-70)")
            ),
            "macd": round(float(macd_line.iloc[-1]), 4),
            "macd_signal": round(float(macd_signal.iloc[-1]), 4),
            "macd_histogram": round(float(macd_hist.iloc[-1]), 4),
            "macd_trend": "Bullish (MACD > Signal)" if float(macd_line.iloc[-1]) > float(macd_signal.iloc[-1]) else "Bearish (MACD < Signal)",
            "atr_14": round(float(atr_14.iloc[-1]), 2),
            "atr_14_pct_price": round((float(atr_14.iloc[-1]) / curr_price) * 100, 2),
            "bollinger_upper_20_2": round(float(bb_upper.iloc[-1]), 2),
            "bollinger_lower_20_2": round(float(bb_lower.iloc[-1]), 2),
            "bollinger_bandwidth_pct": round(float(bb_width.iloc[-1]) * 100, 2),
            "bollinger_position_pct": round(float(bb_position.iloc[-1]) * 100, 2),
            "stoch_k_14": round(float(stoch_k.iloc[-1]), 2),
            "stoch_d_3": round(float(stoch_d.iloc[-1]), 2),
            "momentum_1m_pct": round(float(momentum_21d), 2) if momentum_21d is not None else None,
            "momentum_3m_pct": round(float(momentum_63d), 2) if momentum_63d is not None else None,
            "momentum_6m_pct": round(float(momentum_126d), 2) if momentum_126d is not None else None,
            "momentum_12m_pct": round(float(momentum_252d), 2) if momentum_252d is not None else None,
            "volatility_20d_annualized_pct": round(float(volatility_20d), 2),
            "distance_to_52w_high_pct": round(((curr_price / float(high.tail(252).max())) - 1) * 100, 2),
            "distance_to_52w_low_pct": round(((curr_price / float(low.tail(252).min())) - 1) * 100, 2),
            "daily_return_pct": round(float(returns.iloc[-1]) * 100, 2) if not pd.isna(returns.iloc[-1]) else None,
            "volume_vs_20d_ma_pct": round(float(volume_vs_20d), 2) if volume_vs_20d is not None else None,
        },
        "technical_chart": chart_df.tail(300).to_dict(orient="records"),
    })


def get_stock_fundamentals_raw(ticker: str) -> dict:
    """Fetch full, unmodified fundamentals payloads from FMP stable endpoints."""
    symbol = ticker.strip().upper()
    if not symbol:
        return {"ticker": symbol, "error": "Ticker is required."}

    try:
        payload = {
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
            "historical_price_eod": _fmp_get(
                "/historical-price-eod/full",
                {"symbol": symbol},
                base_url=FMP_STABLE_BASE_URL,
            ),
        }
    except Exception as exc:
        return {"ticker": symbol, "error": f"Failed to fetch fundamentals: {exc}"}

    technical = _build_technical_payload(payload["historical_price_eod"])
    payload["technical_summary"] = technical["technical_summary"]
    payload["technical_chart"] = technical["technical_chart"]
    return _json_safe(payload)


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


def get_street_commentary(limit: int = 30) -> list[dict]:
    """Fetch recent macro commentary items from major street institutions."""
    max_items = max(limit, 0)
    if max_items == 0:
        return []

    entries: list[Any] = []
    per_query_limit = 10
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
