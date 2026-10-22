# Multi-Agent Equity Research

Streamlit app running an automated investment committee debate on US stocks.

```
pip install -r requirements.txt
cp .env.example .env   # set HF_TOKEN + (HF_INFERENCE_ENDPOINT or HF_MODEL) + FMP_API_KEY
streamlit run app.py
```

Workflow in the app:
1. Load a ticker to fetch and display market data/news.
2. Run the investment committee on the loaded ticker dataset.

News ingestion is resilient: it uses direct finance news feeds first and attempts full-article text extraction, with snippet fallback when an article cannot be parsed.

Required data key:
- `FMP_API_KEY` (Financial Modeling Prep API key for fundamentals and news)
