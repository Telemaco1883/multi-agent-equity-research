# Multi-Agent Equity Research

Streamlit app running an automated investment committee debate on US stocks.

```
pip install -r requirements.txt
cp .env.example .env   # set HF_INFERENCE_ENDPOINT and HF_TOKEN
streamlit run app.py
```

Workflow in the app:
1. Load a ticker to fetch and display market data/news.
2. Run the investment committee on the loaded ticker dataset.
