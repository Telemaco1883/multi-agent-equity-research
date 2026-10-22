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


def run_committee(
    ticker: str,
    fundamentals: dict,
    news: list[dict],
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

    notify("analysts", "start")
    jobs = {
        "fundamental": (FUNDAMENTAL_PROMPT, f"Ticker: {ticker}\nFundamentals:\n{fund_json}"),
        "sentiment": (SENTIMENT_PROMPT, f"Ticker: {ticker}\nRecent news:\n{news_text}"),
        "technical": (TECHNICAL_PROMPT, f"Ticker: {ticker}\nTechnical data:\n{tech_json}"),
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
