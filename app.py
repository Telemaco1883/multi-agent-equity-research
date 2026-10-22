"""Streamlit frontend for the Multi-Agent Equity Research Investment Committee."""
from __future__ import annotations

import re

import streamlit as st

import config
from committee import build_report, run_committee
from data_collector import get_latest_news, get_stock_fundamentals

st.set_page_config(page_title="Investment Committee", page_icon="📈", layout="wide")
st.title("Multi-Agent Investment Committee 📈")
st.write(
    "Enter a US stock ticker. Specialist analysts research it, a Portfolio Manager rules, "
    "a Devil's Advocate stress-tests the thesis, and the committee delivers a final verdict."
)

missing = config.missing_config()
if missing:
    st.error("Backend is not configured. Please contact the administrator.")
    st.stop()

COLORS = {"BUY": "green", "HOLD": "orange", "SELL": "red"}


def _fmt_num(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _fmt_currency(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.2f}"


def _render_ticker_data(ticker_symbol: str, fundamentals: dict, news: list[dict]) -> None:
    valuation = fundamentals.get("valuation", {})
    margins = fundamentals.get("margins", {})
    growth = fundamentals.get("growth", {})
    balance_sheet = fundamentals.get("balance_sheet", {})
    technicals = fundamentals.get("technicals", {})

    st.subheader(f"{fundamentals.get('name') or ticker_symbol} ({ticker_symbol})")
    st.caption(
        f"{fundamentals.get('sector') or 'N/A'} | "
        f"{fundamentals.get('industry') or 'N/A'}"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Price", _fmt_currency(fundamentals.get("price")))
    c2.metric("Market Cap", _fmt_currency(fundamentals.get("market_cap")))
    c3.metric("RSI (14)", _fmt_num(technicals.get("rsi_14"), 1))

    c4, c5, c6 = st.columns(3)
    c4.metric("Trailing P/E", _fmt_num(valuation.get("trailing_pe")))
    c5.metric("Forward P/E", _fmt_num(valuation.get("forward_pe")))
    c6.metric("P/S", _fmt_num(valuation.get("price_to_sales")))

    c7, c8, c9 = st.columns(3)
    c7.metric("Revenue Growth", _fmt_pct(growth.get("revenue_growth")))
    c8.metric("Earnings Growth", _fmt_pct(growth.get("earnings_growth")))
    c9.metric("Net Margin", _fmt_pct(margins.get("net")))

    with st.expander("Balance sheet snapshot"):
        st.write(
            {
                "Total cash": _fmt_currency(balance_sheet.get("total_cash")),
                "Total debt": _fmt_currency(balance_sheet.get("total_debt")),
                "Debt/Equity": _fmt_num(balance_sheet.get("debt_to_equity")),
                "Current ratio": _fmt_num(balance_sheet.get("current_ratio")),
                "Free cash flow": _fmt_currency(balance_sheet.get("free_cash_flow")),
            }
        )
    with st.expander("Recent headlines"):
        if news:
            for item in news:
                st.markdown(
                    f"- [{item.get('title', 'Untitled')}]({item.get('link', '')})"
                    f" ({item.get('published', '')})"
                )
                excerpt = (item.get("article_text") or item.get("snippet") or "").strip()
                if excerpt:
                    st.caption(excerpt[:320] + ("..." if len(excerpt) > 320 else ""))
        else:
            st.write("No recent headlines found.")


ticker = st.text_input("Ticker", placeholder="TSLA", key="ticker_input").strip().upper()
if ticker and st.session_state.get("loaded_ticker") and ticker != st.session_state["loaded_ticker"]:
    st.session_state.pop("fundamentals", None)
    st.session_state.pop("news", None)
    st.session_state.pop("result", None)
    st.session_state.pop("loaded_ticker", None)

run = st.button("Run Investment Committee", type="primary")
rendered_in_run = False

if run:
    if not re.fullmatch(r"[A-Z.\-]{1,10}", ticker):
        st.warning("Please enter a valid ticker symbol (e.g., TSLA).")
    else:
        with st.status("Collecting market data and news...", expanded=True) as status:
            st.write("1/5 Fetching fundamentals and recent articles...")
            fundamentals = get_stock_fundamentals(ticker)
            if "error" in fundamentals:
                status.update(label="Data collection failed", state="error")
                st.error(fundamentals["error"])
                st.stop()
            news = get_latest_news(ticker, limit=20)
            if not news:
                st.write("⚠️ No recent news found; sentiment analysis may be limited.")

            st.session_state["fundamentals"] = fundamentals
            st.session_state["news"] = news
            st.session_state["loaded_ticker"] = ticker
            st.session_state.pop("result", None)
            status.update(label="Ticker data loaded", state="complete", expanded=False)

        st.markdown("### Ticker data loaded")
        _render_ticker_data(ticker, fundamentals, news)
        rendered_in_run = True

        try:
            with st.status("Running investment committee...", expanded=True) as status:
                labels = {
                    "analysts": "2/5 Parallel analyst research...",
                    "pm": "3/5 Initial PM thesis...",
                    "devil": "4/5 Devil's advocate challenge...",
                    "final": "5/5 Final committee decision...",
                }
                previews = {
                    "analysts": "Analysts are scoring fundamentals, sentiment, and technicals.",
                    "pm": "PM is synthesizing the first recommendation.",
                    "devil": "Devil's Advocate is stress-testing the thesis.",
                    "final": "PM is finalizing the committee verdict and risk framing.",
                }
                preview_box = st.empty()

                def on_progress(stage: str, state: str, preview: str | None = None) -> None:
                    if state == "start":
                        st.write(labels[stage])
                        preview_box.info(previews[stage])
                    elif state == "done" and preview:
                        preview_box.info(preview)

                result = run_committee(ticker, fundamentals, news, on_progress)
                preview_box.success("Reasoning complete.")
                status.update(label="Committee complete", state="complete", expanded=False)
            st.session_state["result"] = (ticker, result)
        except Exception as exc:
            st.error(f"The committee could not complete: {exc}")

loaded_ticker = st.session_state.get("loaded_ticker", "")
data_loaded_for_ticker = bool(
    loaded_ticker and "fundamentals" in st.session_state and "news" in st.session_state
)

if data_loaded_for_ticker and not rendered_in_run:
    if ticker and ticker != loaded_ticker:
        st.info(f"Showing cached data for {loaded_ticker}. Click run to load {ticker}.")
    _render_ticker_data(loaded_ticker, st.session_state["fundamentals"], st.session_state["news"])

if "result" in st.session_state:
    t, result = st.session_state["result"]
    color = COLORS.get(result["verdict"], "gray")
    st.markdown(f"## {t}: :{color}[{result['verdict']}]")
    st.markdown(f"**Conviction:** {result['conviction']}")
    st.subheader("Executive Summary")
    st.write(result["summary"])
    if result["risks"]:
        st.subheader("Key Risks")
        st.write(result["risks"])

    with st.expander("Fundamental Analyst memo"):
        st.write(result["analysts"]["fundamental"])
    with st.expander("Sentiment Analyst memo"):
        st.write(result["analysts"]["sentiment"])
    with st.expander("Technical Analyst memo"):
        st.write(result["analysts"]["technical"])
    with st.expander("Full debate transcript"):
        st.markdown("### PM Initial Ruling")
        st.write(result["pm_initial"])
        st.markdown("### Devil's Advocate")
        st.write(result["devil"])
        st.markdown("### Final Verdict")
        st.write(result["final"])

    st.download_button(
        "Download report",
        data=build_report(t, result),
        file_name=f"{t}_IC_Report.md",
        mime="text/markdown",
    )
