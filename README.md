# Multi-Agent Equity Research

Streamlit app running an automated investment committee debate on US stocks.

```
pip install -r requirements.txt
cp .env.example .env   # set HF_TOKEN + (HF_INFERENCE_ENDPOINT or HF_MODEL) + FMP_API_KEY
streamlit run app.py
```

Workflow in the app:
1. Pull latest street macro commentary (UBS/BlackRock/Goldman/etc.) and summarize consensus.
2. Load a ticker to fetch and display market data/news.
3. Run the investment committee on the loaded ticker dataset.

News ingestion uses Google News RSS plus article text extraction, with snippet fallback when an article cannot be parsed.

Required data key:
- `FMP_API_KEY` (Financial Modeling Prep API key for fundamentals)
