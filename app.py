"""Streamlit frontend for the Multi-Agent Equity Research Investment Committee."""
from __future__ import annotations

import re

import streamlit as st

import config
from committee import build_report, run_committee
from data_collector import get_latest_news, get_stock_fundamentals

st.set_page_config(page_title="Investment Committee", page_icon="📈", layout="centered")
st.title("📈 Multi-Agent Investment Committee")
st.write(
    "Enter a US stock ticker. Specialist analysts research it, a Portfolio Manager rules, "
    "a Devil's Advocate stress-tests the thesis, and the committee delivers a final verdict."
)

missing = config.missing_config()
if missing:
    st.error("Backend is not configured. Please contact the administrator.")
    st.stop()

COLORS = {"BUY": "green", "HOLD": "orange", "SELL": "red"}

ticker = st.text_input("Ticker", placeholder="TSLA").strip().upper()
run = st.button("Run Investment Committee", type="primary")

if run:
    if not re.fullmatch(r"[A-Z.\-]{1,10}", ticker):
        st.warning("Please enter a valid ticker symbol (e.g., TSLA).")
    else:
        try:
            with st.status("Running investment committee...", expanded=True) as status:
                st.write("1/5 Collecting market data and news...")
                fundamentals = get_stock_fundamentals(ticker)
                if "error" in fundamentals:
                    status.update(label="Data collection failed", state="error")
                    st.error(fundamentals["error"])
                    st.stop()
                news = get_latest_news(ticker)
                if not news:
                    st.write("⚠️ No recent news found; sentiment analysis will be limited.")

                labels = {
                    "analysts": "2/5 Parallel analyst research...",
                    "pm": "3/5 Initial PM thesis...",
                    "devil": "4/5 Devil's advocate challenge...",
                    "final": "5/5 Final committee decision...",
                }

                def on_progress(stage: str, state: str) -> None:
                    if state == "start":
                        st.write(labels[stage])

                result = run_committee(ticker, fundamentals, news, on_progress)
                status.update(label="Committee complete", state="complete", expanded=False)
            st.session_state["result"] = (ticker, result)
        except Exception as exc:
            st.error(f"The committee could not complete: {exc}")

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
