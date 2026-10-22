"""Multi-agent investment committee pipeline."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from llm_client import generate_response

ProgressCallback = Callable[[str, str], None]  # (stage_key, "start" | "done")

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

    def notify(stage: str, status: str) -> None:
        if on_progress:
            on_progress(stage, status)

    news_text = "\n".join(
        f"- {n['title']} ({n['published']}): {n['snippet']}" for n in news
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
    notify("analysts", "done")

    reports = "\n\n".join(f"## {k.title()} Analyst\n{v}" for k, v in analysts.items())

    notify("pm", "start")
    pm_initial = generate_response(PM_PROMPT, f"Ticker: {ticker}\n\n{reports}")
    notify("pm", "done")

    notify("devil", "start")
    devil = generate_response(
        DEVIL_PROMPT, f"Ticker: {ticker}\n\n{reports}\n\n## PM Initial Ruling\n{pm_initial}"
    )
    notify("devil", "done")

    notify("final", "start")
    final = generate_response(
        FINAL_PROMPT,
        f"Ticker: {ticker}\n\n{reports}\n\n## PM Initial Ruling\n{pm_initial}\n\n"
        f"## Devil's Advocate\n{devil}",
    )
    notify("final", "done")

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
