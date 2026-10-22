"""Multi-agent investment committee pipeline."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from llm_client import generate_response

ProgressCallback = Callable[[str, str, str | None], None]  # (stage_key, "start" | "done", preview)

FUNDAMENTAL_PROMPT = (
    "You are a Fundamental Analyst on an investment committee. Evaluate valuation, balance sheet "
    "strength, margins and earnings trajectory using ONLY the data provided. Be concise and "
    "quantitative; note missing data explicitly."
)
SENTIMENT_PROMPT = (
    "You are a Sentiment Analyst on an investment committee. From the recent news, identify key "
    "narrative themes, risks, catalysts and overall media sentiment. Be concise and cite headlines."
)
TECHNICAL_PROMPT = (
    "You are a Technical Analyst on an investment committee. Evaluate trend, momentum indicators "
    "(50/200-day MAs, RSI) and key support/resistance levels using ONLY the data provided."
)
MACRO_CONSENSUS_PROMPT = """
You are a Senior Global Macro Strategist at an institutional asset management firm.
Synthesize the prevailing Street consensus across major asset classes based SOLELY on the provided market commentary and article snippets.

### INSTRUCTIONS & CONSTRAINTS:
1. FOCUS: Synthesize macro/cross-asset views (rate trajectory, central bank policy, macro regime, asset allocation tilts). Ignore single-stock earnings, corporate PR, and retail bank product news.
2. STRICT GROUNDING: Rely exclusively on substantive snippet text. Do NOT rely on isolated headlines. Do NOT extrapolate or introduce external market priors.
3. FORMAT INTEGRITY: Output EXACTLY the 5 sections below. Do not add intro/outro text.
4. EVIDENCE THRESHOLD: If a category lacks concrete commentary in the snippets, state exactly: "Consensus Stance: Insufficient Evidence | Details: No substantive data in provided feeds."

### OUTPUT TEMPLATE:
* **Rates & Inflation**: [Consensus Stance: Dovish / Neutral / Hawkish] — Summarize expected rate path, terminal rate/cuts, inflation trajectory, and duration guidance with supporting details.
* **Equities**: [Consensus Stance: Bullish / Neutral / Bearish] — Summarize market bias (Overweight / Neutral / Underweight), major drivers (growth vs. multiple compression), and regional/sector tilts mentioned.
* **Credit & Spreads**: [Consensus Stance: Risk-On / Neutral / Risk-Off] — Summarize IG/HY spread direction, refinancing/default concerns, or credit appetite.
* **FX & Commodities**: [Consensus Stance: USD Strong/Range/Weak | Commodities Bull/Bear] — Summarize Dollar direction, energy/crude outlook, or gold/cyclical metals views.
* **Positioning & Risk Sentiment**: [Consensus Stance: Crowded / Neutral / Defensive] — Summarize broad institutional positioning, cash balances, and primary macro tail risks cited (e.g., tariffs, stagflation, fiscal supply).
"""
PM_PROMPT = (
    "You are the Portfolio Manager chairing an investment committee. Synthesize the analyst reports "
    "into an initial recommendation. Start with exactly two lines:\n"
    "VERDICT: BUY|HOLD|SELL\nCONVICTION: High|Medium|Low\n"
    "Then give the key rationale."
)
DEVIL_PROMPT = (
    "You are the Devil's Advocate. Take the OPPOSITE position to the PM's recommendation. Highlight "
    "blind spots, downside tail risks and challenge the PM's assumptions with specifics."
)
FINAL_PROMPT = (
    "You are the Portfolio Manager delivering the final Investment Committee verdict after reviewing "
    "the Devil's Advocate challenge. Start with exactly two lines:\n"
    "VERDICT: BUY|HOLD|SELL\nCONVICTION: High|Medium|Low\n"
    "Then provide these sections: Executive Summary, Horizon, Sizing Guidance, Key Risks to Monitor "
    "(bulleted)."
)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float))


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:,.3f}" if abs(value) < 100 else f"{value:,.2f}"
    return str(value)


def _fundamentals_completeness(fundamentals: dict[str, Any]) -> float:
    keys = [
        ("valuation", "trailing_pe"),
        ("valuation", "forward_pe"),
        ("valuation", "ev_to_ebitda"),
        ("valuation", "price_to_sales"),
        ("valuation", "peg_ratio"),
        ("margins", "gross"),
        ("margins", "operating"),
        ("margins", "net"),
        ("growth", "revenue_growth"),
        ("growth", "earnings_growth"),
        ("balance_sheet", "total_cash"),
        ("balance_sheet", "total_debt"),
        ("balance_sheet", "debt_to_equity"),
        ("balance_sheet", "current_ratio"),
        ("balance_sheet", "free_cash_flow"),
    ]
    present = 0
    for section, key in keys:
        if fundamentals.get(section, {}).get(key) is not None:
            present += 1
    return present / len(keys)


def _deterministic_fundamental_memo(ticker: str, fundamentals: dict[str, Any]) -> str:
    v = fundamentals.get("valuation", {})
    m = fundamentals.get("margins", {})
    g = fundamentals.get("growth", {})
    b = fundamentals.get("balance_sheet", {})
    lines = [
        f"Fundamental view for {ticker} (data-backed summary):",
        "",
        "Valuation:",
        f"- Trailing P/E: {_fmt(v.get('trailing_pe'))}",
        f"- Forward P/E: {_fmt(v.get('forward_pe'))}",
        f"- EV/EBITDA: {_fmt(v.get('ev_to_ebitda'))}",
        f"- Price/Sales: {_fmt(v.get('price_to_sales'))}",
        f"- PEG Ratio: {_fmt(v.get('peg_ratio'))}",
        "",
        "Profitability:",
        f"- Gross margin: {_fmt(m.get('gross'))}",
        f"- Operating margin: {_fmt(m.get('operating'))}",
        f"- Net margin: {_fmt(m.get('net'))}",
        "",
        "Growth:",
        f"- Revenue growth: {_fmt(g.get('revenue_growth'))}",
        f"- Earnings growth: {_fmt(g.get('earnings_growth'))}",
        "",
        "Balance sheet / cash:",
        f"- Total cash: {_fmt(b.get('total_cash'))}",
        f"- Total debt: {_fmt(b.get('total_debt'))}",
        f"- Debt/Equity: {_fmt(b.get('debt_to_equity'))}",
        f"- Current ratio: {_fmt(b.get('current_ratio'))}",
        f"- Free cash flow: {_fmt(b.get('free_cash_flow'))}",
    ]
    return "\n".join(lines)


def _fundamental_response_is_incoherent(response: str, fundamentals: dict[str, Any]) -> bool:
    if not response:
        return True
    lower = response.lower()
    available = _fundamentals_completeness(fundamentals)
    if available < 0.45:
        return False
    missing_hits = lower.count("missing data")
    if missing_hits >= 3:
        return True
    if "without sufficient data" in lower and available >= 0.6:
        return True
    return False


def parse_verdict(text: str) -> tuple[str, str]:
    """Extract (verdict, conviction) from PM output, defaulting to HOLD / Low."""
    verdict = re.search(r"VERDICT\W*(BUY|HOLD|SELL)", text, re.IGNORECASE)
    conviction = re.search(r"CONVICTION\W*(High|Medium|Low)", text, re.IGNORECASE)
    return (
        verdict.group(1).upper() if verdict else "HOLD",
        conviction.group(1).capitalize() if conviction else "Low",
    )


def extract_section(text: str, heading: str) -> str:
    """Return the body of a named section (up to the next heading), or ''."""
    pattern = rf"(?:^|\n)[#*\s]*{re.escape(heading)}[^\n]*\n(.*?)(?=\n[#*\s]*(?:Executive Summary|Horizon|Sizing|Key Risks)|\Z)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _first_sentences(text: str, max_sentences: int = 2) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]
    if not parts:
        return cleaned[:280]
    preview = " ".join(parts[:max_sentences]).strip()
    return preview[:360]


def summarize_street_consensus(commentary: list[dict[str, Any]]) -> str:
    """Create a concise macro consensus summary from street commentary items."""
    if not commentary:
        return "No major street macro commentary was found in the latest feed."
    payload = "\n\n".join(
        (
            f"Headline: {item.get('title', '')}\n"
            f"Published: {item.get('published', '')}\n"
            f"Source: {item.get('link', '')}\n"
            f"Content: {(item.get('article_text') or item.get('snippet') or '')[:900]}"
        )
        for item in commentary[:60]
    )
    try:
        summary = generate_response(MACRO_CONSENSUS_PROMPT, payload, max_tokens=450)
        if summary and summary.strip():
            return summary.strip()
    except Exception:
        pass

    fallback_lines = []
    for item in commentary[:5]:
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        if title:
            fallback_lines.append(f"- {title}")
            if snippet:
                fallback_lines.append(f"  {_first_sentences(snippet, 1)}")
    return "\n".join(fallback_lines) if fallback_lines else "No macro consensus summary available."


def run_committee(
    ticker: str,
    fundamentals: dict,
    news: list[dict],
    macro_summary: str = "",
    macro_commentary: list[dict[str, Any]] | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the 4-stage committee and return all outputs.

    Keys: analysts (fundamental/sentiment/technical), pm_initial, devil, final,
    verdict, conviction, summary, risks.
    """

    def notify(stage: str, status: str, preview: str | None = None) -> None:
        if on_progress:
            on_progress(stage, status, preview)

    news_text = "\n\n".join(
        (
            f"Headline: {n.get('title', '')}\n"
            f"Published: {n.get('published', '')}\n"
            f"Source: {n.get('link', '')}\n"
            f"Content: {(n.get('article_text') or n.get('snippet') or '')[:1200]}"
        )
        for n in news
    ) or "No recent news available."
    fund_json = json.dumps(
        {k: v for k, v in fundamentals.items() if k != "technicals"}, indent=2, default=str
    )
    tech_json = json.dumps(
        {"price": fundamentals.get("price"), **fundamentals.get("technicals", {})}, indent=2, default=str
    )
    macro_items_text = "\n\n".join(
        (
            f"Headline: {item.get('title', '')}\n"
            f"Published: {item.get('published', '')}\n"
            f"Source: {item.get('link', '')}\n"
            f"Content: {(item.get('article_text') or item.get('snippet') or '')[:800]}"
        )
        for item in (macro_commentary or [])[:8]
    )
    macro_context = (
        f"Street macro consensus:\n{macro_summary or 'N/A'}\n\nStreet macro commentary items:\n"
        f"{macro_items_text or 'N/A'}"
    )

    notify("analysts", "start")
    jobs = {
        "fundamental": (
            FUNDAMENTAL_PROMPT,
            f"Ticker: {ticker}\n\n{macro_context}\n\nFundamentals:\n{fund_json}",
        ),
        "sentiment": (
            SENTIMENT_PROMPT,
            f"Ticker: {ticker}\n\n{macro_context}\n\nRecent news:\n{news_text}",
        ),
        "technical": (
            TECHNICAL_PROMPT,
            f"Ticker: {ticker}\n\n{macro_context}\n\nTechnical data:\n{tech_json}",
        ),
    }
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {k: pool.submit(generate_response, s, u) for k, (s, u) in jobs.items()}
        analysts = {k: f.result() for k, f in futures.items()}
    if _fundamental_response_is_incoherent(analysts["fundamental"], fundamentals):
        analysts["fundamental"] = _deterministic_fundamental_memo(ticker, fundamentals)
    analysts_preview = "\n".join(
        [
            f"**Fundamental:** {_first_sentences(analysts['fundamental'])}",
            f"**Sentiment:** {_first_sentences(analysts['sentiment'])}",
            f"**Technical:** {_first_sentences(analysts['technical'])}",
        ]
    )
    notify("analysts", "done", analysts_preview)

    reports = "\n\n".join(f"## {k.title()} Analyst\n{v}" for k, v in analysts.items())

    notify("pm", "start")
    pm_initial = generate_response(PM_PROMPT, f"Ticker: {ticker}\n\n{reports}")
    notify("pm", "done", _first_sentences(pm_initial))

    notify("devil", "start")
    devil = generate_response(
        DEVIL_PROMPT, f"Ticker: {ticker}\n\n{reports}\n\n## PM Initial Ruling\n{pm_initial}"
    )
    notify("devil", "done", _first_sentences(devil))

    notify("final", "start")
    final = generate_response(
        FINAL_PROMPT,
        f"Ticker: {ticker}\n\n{reports}\n\n## PM Initial Ruling\n{pm_initial}\n\n"
        f"## Devil's Advocate\n{devil}",
    )
    notify("final", "done", _first_sentences(final))

    verdict, conviction = parse_verdict(final)
    return {
        "analysts": analysts,
        "pm_initial": pm_initial,
        "devil": devil,
        "final": final,
        "verdict": verdict,
        "conviction": conviction,
        "summary": extract_section(final, "Executive Summary") or final,
        "risks": extract_section(final, "Key Risks"),
    }


def build_report(ticker: str, result: dict[str, Any]) -> str:
    """Assemble the full Markdown report (verdict, research, debate transcript)."""
    a = result["analysts"]
    return (
        f"# {ticker} Investment Committee Report\n\n"
        f"**Verdict:** {result['verdict']}  \n**Conviction:** {result['conviction']}\n\n"
        f"## Final Committee Decision\n{result['final']}\n\n"
        f"## Analyst Research\n### Fundamental Analyst\n{a['fundamental']}\n\n"
        f"### Sentiment Analyst\n{a['sentiment']}\n\n### Technical Analyst\n{a['technical']}\n\n"
        f"## Debate Transcript\n### PM Initial Ruling\n{result['pm_initial']}\n\n"
        f"### Devil's Advocate\n{result['devil']}\n\n### Final Verdict\n{result['final']}\n"
    )
