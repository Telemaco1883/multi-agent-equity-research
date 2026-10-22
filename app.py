"""Streamlit frontend for the Multi-Agent Equity Research Investment Committee."""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

import config
from committee import build_report, run_committee, summarize_street_consensus
from data_collector import get_latest_news, get_stock_fundamentals, get_street_commentary

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

COLORS = {"BUY": "green", "NEUTRAL": "orange", "SELL": "red"}

def _render_ticker_data(
    ticker_symbol: str,
    fundamentals: dict,
    news: list[dict],
    macro_summary: str,
    macro_commentary: list[dict],
    result: dict | None = None,
) -> None:
    with st.expander("1) Street macro consensus", expanded=True):
        st.write(macro_summary or "No street macro commentary summary available.")
        if macro_commentary:
            with st.expander("Street macro source headlines", expanded=False):
                for item in macro_commentary:
                    st.markdown(
                        f"- [{item.get('title', 'Untitled')}]({item.get('link', '')})"
                        f" ({item.get('published', '')})"
                    )

    with st.expander("2) Raw downloaded data", expanded=True):
        st.markdown("**Technical chart (price + moving averages)**")
        technical_chart = fundamentals.get("technical_chart") or []
        if technical_chart:
            chart_df = pd.DataFrame(technical_chart)
            if not chart_df.empty and "date" in chart_df.columns:
                chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
                chart_df = chart_df.dropna(subset=["date"]).set_index("date")
                line_cols = [col for col in ["close", "sma_20", "sma_50", "sma_100", "sma_200", "ema_21", "ema_50"] if col in chart_df.columns]
                if line_cols:
                    st.line_chart(chart_df[line_cols], use_container_width=True)
                osc_cols = [col for col in ["rsi_14"] if col in chart_df.columns]
                if osc_cols:
                    st.markdown("**RSI (14)**")
                    st.line_chart(chart_df[osc_cols], use_container_width=True)
                macd_cols = [col for col in ["macd", "macd_signal", "macd_hist"] if col in chart_df.columns]
                if macd_cols:
                    st.markdown("**MACD**")
                    st.line_chart(chart_df[macd_cols], use_container_width=True)
        tech_summary = fundamentals.get("technical_summary")
        if tech_summary:
            st.markdown("**Technical summary**")
            st.json(tech_summary)

        st.markdown("**Fundamentals JSON**")
        st.json(fundamentals)
        st.markdown("**Recent headlines**")
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

    if not result:
        return

    with st.expander("3) Fundamental analyst", expanded=True):
        st.write(result["analysts"]["fundamental"])
    with st.expander("4) Sentiment analyst", expanded=False):
        st.write(result["analysts"]["sentiment"])
    with st.expander("5) Technical analyst", expanded=False):
        st.write(result["analysts"]["technical"])
    with st.expander("6) PM initial ruling", expanded=False):
        st.write(result["pm_initial"])
    with st.expander("7) Devil's advocate", expanded=False):
        st.write(result["devil"])
    with st.expander("8) Investment committee decision", expanded=True):
        color = COLORS.get(result["verdict"], "gray")
        st.markdown(f"**Verdict:** :{color}[{result['verdict']}]")
        st.write(result["final"])


ticker = st.text_input("Ticker", placeholder="TSLA", key="ticker_input").strip().upper()
if ticker and st.session_state.get("loaded_ticker") and ticker != st.session_state["loaded_ticker"]:
    st.session_state.pop("fundamentals", None)
    st.session_state.pop("news", None)
    st.session_state.pop("macro_commentary", None)
    st.session_state.pop("macro_summary", None)
    st.session_state.pop("result", None)
    st.session_state.pop("loaded_ticker", None)

run = st.button("Run Investment Committee", type="primary")
rendered_in_run = False

cached_fundamentals = st.session_state.get("fundamentals")
if isinstance(cached_fundamentals, dict) and "profile" not in cached_fundamentals:
    st.session_state.pop("fundamentals", None)
    st.session_state.pop("result", None)

if run:
    if not re.fullmatch(r"[A-Z.\-]{1,10}", ticker):
        st.warning("Please enter a valid ticker symbol (e.g., TSLA).")
    else:
        with st.status("Collecting market data and news...", expanded=True) as status:
            st.write("1/6 Fetching latest street macro commentary...")
            macro_commentary = get_street_commentary(limit=30)
            macro_summary = summarize_street_consensus(macro_commentary)
            st.session_state["macro_commentary"] = macro_commentary
            st.session_state["macro_summary"] = macro_summary
            if not macro_commentary:
                st.write("⚠️ No street macro commentary found; consensus summary may be limited.")

            st.write("2/6 Fetching fundamentals and recent articles...")
            fundamentals = get_stock_fundamentals(ticker)
            if "error" in fundamentals:
                status.update(label="Data collection failed", state="error")
                st.error(fundamentals["error"])
            else:
                news = get_latest_news(ticker, limit=20)
                if not news:
                    st.write("⚠️ No recent news found; sentiment analysis may be limited.")

                st.session_state["fundamentals"] = fundamentals
                st.session_state["news"] = news
                st.session_state["loaded_ticker"] = ticker
                st.session_state.pop("result", None)
                status.update(label="Data fetch complete", state="complete", expanded=False)

        if "error" in fundamentals:
            with st.expander("Street macro consensus (Step 1)", expanded=True):
                st.write(macro_summary or "No street macro commentary summary available.")
                if macro_commentary:
                    with st.expander("Street macro source headlines", expanded=False):
                        for item in macro_commentary:
                            st.markdown(
                                f"- [{item.get('title', 'Untitled')}]({item.get('link', '')})"
                                f" ({item.get('published', '')})"
                            )
        else:
            try:
                with st.status("Running investment committee...", expanded=True) as status:
                    labels = {
                        "analysts": "3/6 Parallel analyst research...",
                        "pm": "4/6 Initial PM thesis...",
                        "devil": "5/6 Devil's advocate challenge...",
                        "final": "6/6 Final committee decision...",
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

                    result = run_committee(
                        ticker,
                        fundamentals,
                        news,
                        macro_summary=macro_summary,
                        macro_commentary=macro_commentary,
                        on_progress=on_progress,
                    )
                    preview_box.success("Reasoning complete.")
                    status.update(label="Committee complete", state="complete", expanded=False)
                st.session_state["result"] = (ticker, result)
                _render_ticker_data(ticker, fundamentals, news, macro_summary, macro_commentary, result)
                rendered_in_run = True
            except Exception as exc:
                st.error(f"The committee could not complete: {exc}")

loaded_ticker = st.session_state.get("loaded_ticker", "")
data_loaded_for_ticker = bool(
    loaded_ticker
    and "fundamentals" in st.session_state
    and "news" in st.session_state
    and "macro_summary" in st.session_state
)

if data_loaded_for_ticker and not rendered_in_run:
    if ticker and ticker != loaded_ticker:
        st.info(f"Showing cached data for {loaded_ticker}. Click run to load {ticker}.")
    _render_ticker_data(
        loaded_ticker,
        st.session_state["fundamentals"],
        st.session_state["news"],
        st.session_state.get("macro_summary", ""),
        st.session_state.get("macro_commentary", []),
        st.session_state.get("result", ("", None))[1]
        if st.session_state.get("result", ("", None))[0] == loaded_ticker
        else None,
    )
if "result" in st.session_state:
    t, result = st.session_state["result"]
    st.download_button(
        "Download report",
        data=build_report(t, result),
        file_name=f"{t}_IC_Report.md",
        mime="text/markdown",
    )
