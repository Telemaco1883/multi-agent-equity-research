"""Multi-agent investment committee pipeline."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from llm_client import generate_response

ProgressCallback = Callable[[str, str, str | None], None]  # (stage_key, "start" | "done", preview)
MAX_INPUT_CHARS = 30_000

FUNDAMENTAL_PROMPT = """
You are the Fundamental Analyst on an investment committee.
Use ONLY the provided raw fundamentals JSON and recent news context.
Reason deeply, but output only the final concise answer.

Rules:
1. No invented numbers, no external data.
2. Every numeric claim must be tied to explicit fields from the input.
3. If a field is absent, say "Not available in provided data."
4. Keep the output under 220 words.

Output exactly this structure:
## Fundamental Analyst
**Stance:** Bullish | Neutral | Bearish
**What matters most (3 bullets):**
- ...
- ...
- ...
**Evidence from provided data (4 bullets max):**
- [field/value] -> implication
**Primary failure mode (1 bullet):**
- ...
"""
SENTIMENT_PROMPT = """
You are the Sentiment Analyst on an investment committee.
Use ONLY the provided recent news and article content.
Reason deeply, but output only the final concise answer.

Rules:
1. Do not infer facts not present in the supplied news content.
2. Separate tone from hard catalysts/risks.
3. Keep output under 200 words.

Output exactly this structure:
## Sentiment Analyst
**Stance:** Risk-On | Neutral | Risk-Off
**Dominant narratives (3 bullets):**
- ...
- ...
- ...
**Near-term catalysts (2 bullets):**
- ...
- ...
**Sentiment risks (2 bullets):**
- ...
- ...
"""
TECHNICAL_PROMPT = """
You are the Technical Analyst on an investment committee.
Use ONLY the provided technical summary JSON.
Reason deeply, but output only the final concise answer.

Rules:
1. Use indicators directly from technical summary (RSI, MA/EMA, MACD, momentum, ATR, Bollinger, stochastic, volatility) when present.
2. If technical fields are missing, state that explicitly.
3. Do not use profile/fundamental/news fields for technical conclusions.
4. Do not output any confidence/rating field.
5. Do not fabricate support/resistance levels.
6. Keep output under 180 words.

Output exactly this structure:
## Technical Analyst
**Stance:** Bullish | Neutral | Bearish
**Trend/Momentum read (3 bullets):**
- ...
- ...
- ...
**Data quality note (1 bullet):**
- ...
"""
MACRO_CONSENSUS_PROMPT = """
You are a Senior Global Macro Strategist at an institutional asset management firm.
Synthesize the prevailing Street consensus across major asset classes based SOLELY on the provided market commentary and article content.

### INSTRUCTIONS & CONSTRAINTS:
1. FOCUS: Synthesize macro/cross-asset views (rate trajectory, central bank policy, macro regime, asset allocation tilts). Ignore single-stock earnings, corporate PR, and retail bank product news.
2. STRICT GROUNDING: Rely exclusively on substantive article content. Do NOT rely on isolated headlines. Do NOT extrapolate or introduce external market priors.
3. FORMAT INTEGRITY: Output EXACTLY the 5 sections below. Do not add intro/outro text.
4. EVIDENCE THRESHOLD: If a category lacks concrete commentary, state exactly: "Consensus Stance: Insufficient Evidence | Details: No substantive market commentary available for this category."

### OUTPUT TEMPLATE:
* **Rates & Inflation**: [Consensus Stance: Dovish / Neutral / Hawkish] — Summarize expected rate path, terminal rate/cuts, inflation trajectory, and duration guidance with supporting details.
* **Equities**: [Consensus Stance: Bullish / Neutral / Bearish] — Summarize market bias (Overweight / Neutral / Underweight), major drivers (growth vs. multiple compression), and regional/sector tilts mentioned.
* **Credit & Spreads**: [Consensus Stance: Risk-On / Neutral / Risk-Off] — Summarize IG/HY spread direction, refinancing/default concerns, or credit appetite.
* **FX & Commodities**: [Consensus Stance: USD Strong/Range/Weak | Commodities Bull/Bear] — Summarize Dollar direction, energy/crude outlook, or gold/cyclical metals views.
* **Positioning & Risk Sentiment**: [Consensus Stance: Crowded / Neutral / Defensive] — Summarize broad institutional positioning, cash balances, and primary macro tail risks cited (e.g., tariffs, stagflation, fiscal supply).
"""
PM_PROMPT = (
    "You are the Portfolio Manager chairing an investment committee. Synthesize the analyst reports "
    "into an initial recommendation using ONLY provided content. Think deeply, but output only final result.\n"
    "HOLD/NEUTRAL are invalid for this stage.\n"
    "Numeric style rule: when citing numbers, round to max 2 decimals; for large currency values use $B/$M units.\n"
    "Output EXACTLY this format:\n"
    "VERDICT: BUY|SELL\n"
    "THESIS:\n"
    "- <bullet 1>\n"
    "- <bullet 2>\n"
    "- <bullet 3>\n"
    "- <bullet 4>\n"
    "- <bullet 5>\n"
)
DEVIL_PROMPT = (
    "You are the Devil's Advocate. Defend the forced opposite verdict with the strongest possible case "
    "using ONLY provided content. Think deeply, but output only final result.\n"
    "Numeric style rule: when citing numbers, round to max 2 decimals; for large currency values use $B/$M units.\n"
    "Output EXACTLY this format:\n"
    "VERDICT: BUY|SELL\n"
    "COUNTER-THESIS:\n"
    "- <bullet 1>\n"
    "- <bullet 2>\n"
    "- <bullet 3>\n"
    "CRITICAL ASSUMPTION FAILURES:\n"
    "- <bullet 1>\n"
    "- <bullet 2>\n"
)
FINAL_PROMPT = (
    "You are the Portfolio Manager delivering the final Investment Committee verdict after reviewing "
    "the Devil's Advocate challenge. Use only provided content and output a structured final decision.\n"
    "Numeric style rule: when citing numbers, round to max 2 decimals; for large currency values use $B/$M units.\n"
    "Output EXACTLY this format:\n"
    "VERDICT: BUY|SELL|NEUTRAL\n"
    "Executive Summary\n"
    "- <bullet 1>\n"
    "- <bullet 2>\n"
    "Horizon\n"
    "- <short/medium/long guidance>\n"
    "Sizing Guidance\n"
    "- <position sizing guidance>\n"
    "Key Risks\n"
    "- <risk 1>\n"
    "- <risk 2>\n"
)


def parse_verdict(text: str) -> str:
    """Extract verdict from final output, defaulting to NEUTRAL."""
    verdict = re.search(r"VERDICT\W*(BUY|SELL|NEUTRAL|HOLD)", text, re.IGNORECASE)
    verdict_value = verdict.group(1).upper() if verdict else "NEUTRAL"
    if verdict_value == "HOLD":
        verdict_value = "NEUTRAL"
    return verdict_value


def _parse_pm_verdict(text: str) -> str:
    verdict = re.search(r"VERDICT\W*(BUY|SELL)", text, re.IGNORECASE)
    return verdict.group(1).upper() if verdict else "BUY"

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


def _pack_full_items_with_budget(items: list[str], max_chars: int) -> str:
    """Pack whole items without truncating each item; stop when max total chars would be exceeded."""
    packed: list[str] = []
    used = 0
    for item in items:
        text = (item or "").strip()
        if not text:
            continue
        add = len(text) + (2 if packed else 0)
        if used + add > max_chars:
            break
        packed.append(text)
        used += add
    return "\n\n".join(packed)


def _llm_fundamentals_json(fundamentals: dict[str, Any]) -> str:
    """Pack full sections (no per-section truncation) up to a safe total prompt budget."""
    sections = []
    ordered_keys = [
        "profile",
        "income_statement",
        "balance_sheet_statement",
        "cash_flow_statement",
        "ratios_ttm",
        "key_metrics_ttm",
        "technical_summary",
    ]
    for key in ordered_keys:
        if key in fundamentals:
            section_json = json.dumps(fundamentals[key], indent=2, default=str)
            sections.append(f"{key}:\n{section_json}")
    packed = _pack_full_items_with_budget(sections, max_chars=12_000)
    return packed or "{}"


def summarize_street_consensus(commentary: list[dict[str, Any]]) -> str:
    """Create a concise macro consensus summary from street commentary items."""
    if not commentary:
        return "No major street macro commentary was found in the latest feed."
    macro_items = [
        (
            f"Headline: {item.get('title', '')}\n"
            f"Published: {item.get('published', '')}\n"
            f"Source: {item.get('link', '')}\n"
            f"Content: {(item.get('article_text') or item.get('snippet') or '')}"
        )
        for item in commentary
    ]
    payload = _pack_full_items_with_budget(macro_items, max_chars=MAX_INPUT_CHARS - 6_000)
    try:
        summary = generate_response(MACRO_CONSENSUS_PROMPT, payload, max_tokens=800)
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
    verdict, summary, risks.
    """

    def notify(stage: str, status: str, preview: str | None = None) -> None:
        if on_progress:
            on_progress(stage, status, preview)

    news_items = [
        (
            f"Headline: {n.get('title', '')}\n"
            f"Published: {n.get('published', '')}\n"
            f"Source: {n.get('link', '')}\n"
            f"Content: {(n.get('article_text') or n.get('snippet') or '')}"
        )
        for n in news
    ]
    news_text = _pack_full_items_with_budget(news_items, max_chars=8_000) or "No recent news available."
    full_fundamentals_json = _llm_fundamentals_json(fundamentals)
    technical_summary_json = json.dumps(fundamentals.get("technical_summary", {}), indent=2, default=str)
    macro_items = [
        (
            f"Headline: {item.get('title', '')}\n"
            f"Published: {item.get('published', '')}\n"
            f"Source: {item.get('link', '')}\n"
            f"Content: {(item.get('article_text') or item.get('snippet') or '')}"
        )
        for item in (macro_commentary or [])
    ]
    macro_items_text = _pack_full_items_with_budget(macro_items, max_chars=8_000)
    macro_context = (
        f"Street macro consensus:\n{macro_summary or 'N/A'}\n\nStreet macro commentary items:\n"
        f"{macro_items_text or 'N/A'}"
    )

    notify("analysts", "start")
    jobs = {
        "fundamental": (
            FUNDAMENTAL_PROMPT,
            f"Ticker: {ticker}\n\n{macro_context}\n\nFull fundamentals JSON:\n{full_fundamentals_json}",
            600,
        ),
        "sentiment": (
            SENTIMENT_PROMPT,
            f"Ticker: {ticker}\n\n{macro_context}\n\nFull fundamentals JSON:\n{full_fundamentals_json}\n\nRecent news:\n{news_text}",
            600,
        ),
        "technical": (
            TECHNICAL_PROMPT,
            f"Ticker: {ticker}\n\n{macro_context}\n\nTechnical summary JSON:\n{technical_summary_json}",
            600,
        ),
    }
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            k: pool.submit(generate_response, s, u, max_tokens=max_tokens)
            for k, (s, u, max_tokens) in jobs.items()
        }
        analysts = {k: f.result() for k, f in futures.items()}
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
    pm_initial = generate_response(
        PM_PROMPT,
        f"Ticker: {ticker}\n\n{macro_context}\n\nFull fundamentals JSON:\n{full_fundamentals_json}\n\n{reports}",
        max_tokens=500,
    )
    pm_verdict = _parse_pm_verdict(pm_initial)
    notify("pm", "done", _first_sentences(pm_initial))
    required_opposite = "SELL" if pm_verdict == "BUY" else "BUY"

    notify("devil", "start")
    devil = generate_response(
        DEVIL_PROMPT,
        f"Ticker: {ticker}\n\n{macro_context}\n\nFull fundamentals JSON:\n{full_fundamentals_json}\n\n"
        f"You MUST defend this opposite verdict: {required_opposite}.\n\n"
        f"{reports}\n\n## PM Initial Ruling\n{pm_initial}",
        max_tokens=600,
    )
    notify("devil", "done", _first_sentences(devil))

    notify("final", "start")
    final = generate_response(
        FINAL_PROMPT,
        f"Ticker: {ticker}\n\n{macro_context}\n\nFull fundamentals JSON:\n{full_fundamentals_json}\n\n"
        f"{reports}\n\n## PM Initial Ruling\n{pm_initial}\n\n"
        f"## Devil's Advocate\n{devil}",
        max_tokens=900,
    )
    notify("final", "done", _first_sentences(final))

    verdict = parse_verdict(final)
    return {
        "analysts": analysts,
        "pm_initial": pm_initial,
        "devil": devil,
        "final": final,
        "verdict": verdict,
        "summary": extract_section(final, "Executive Summary") or final,
        "risks": extract_section(final, "Key Risks"),
    }


def build_report(ticker: str, result: dict[str, Any]) -> str:
    """Assemble the full Markdown report (verdict, research, debate transcript)."""
    a = result["analysts"]
    return (
        f"# {ticker} Investment Committee Report\n\n"
        f"**Verdict:** {result['verdict']}\n\n"
        f"## Final Committee Decision\n{result['final']}\n\n"
        f"## Analyst Research\n### Fundamental Analyst\n{a['fundamental']}\n\n"
        f"### Sentiment Analyst\n{a['sentiment']}\n\n### Technical Analyst\n{a['technical']}\n\n"
        f"## Debate Transcript\n### PM Initial Ruling\n{result['pm_initial']}\n\n"
        f"### Devil's Advocate\n{result['devil']}\n\n### Final Verdict\n{result['final']}\n"
    )
