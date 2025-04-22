# 💼 FinAgent: Agentic RAG Chatbot for Financial Insight

A GenAI-powered Retrieval-Augmented Generation (RAG) chatbot with LangChain agents that can reason over financial documents, query structured stock data, and analyze macroeconomic indicators. Inspired by real-world GenAI applications in enterprise, this project combines document Q&A with tool-augmented reasoning for deep financial analysis.

---

## 🧠 Features

- 🔍 **Document RAG**: Semantic search & Q&A on SEC 10-K/10-Q filings
- 📈 **Structured Querying**: SQL-powered analysis on stock prices and fundamentals
- 🧰 **Agent Tools**:
  - `FREDTool` — for macroeconomic data (GDP, CPI, rates)
  - `YahooTool` — for historical stock metrics and ratios
  - `SECParserTool` — for parsing financial PDFs or HTML filings
  - `PlotTool` — auto-generates line/bar charts for financial data
- 🤖 **LangChain Agent (ReAct)**: Selects and chains tools to fulfill multi-step queries
- ��️ **UI Interface**: Streamlit frontend for interactive use

---

## 📁 Folder Structure

```
finance-agent/
├── data/
│   ├── sec_filings/              # SEC 10-Ks and investor PDFs
│   ├── market_data/              # yfinance CSVs
│   ├── plots/                    # Generated visualizations
│   └── fred/                     # Cached macroeconomic series
│
├── tools/
│   ├── fred_tool.py              # FRED API wrapper
│   ├── yahoo_tool.py             # Yahoo Finance tool
│   ├── sec_tool.py               # HTML/PDF parsing for filings
│   └── plot_tool.py              # matplotlib-based plot tool
│
├── agents/
│   └── finance_agent.py          # LangChain agent logic
│
├── vectorstore/
│   ├── index_sec.py              # Embedding logic
│   └── chroma_db/                # Stored vector DB
│
├── sql/
│   ├── db_init.py                # Load structured data to SQLite
│   └── db.sqlite3                # Financial tables: price, ratios, fundamentals
│
├── rag/
│   └── rag_chain.py              # Retriever + LLM Q&A
│
├── ui/
│   └── app.py                    # Streamlit frontend
│
├── .env                          # API keys
├── requirements.txt
└── README.md
```

---

## 🔄 Example Queries

- "Summarize Tesla's risk factors from the latest 10-K."
- "What's Apple's 5-year average free cash flow?"
- "Compare S&P 500 performance vs interest rate hikes."
- "Show CPI vs Nasdaq 100 index for the last 2 years."
- "Visualize my portfolio Sharpe ratio trend over 2020–2023."

---

## 🔑 API Keys Required

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-...
FRED_API_KEY=your_fred_api_key
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/finance-agent.git
cd finance-agent
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/sec_filings data/plots data/market_data data/fred
```

### Prepare Data (Optional)
```bash
# Index sample SEC filings (if available)
python vectorstore/index_sec.py

# Initialize SQLite database with sample data
python sql/db_init.py
```

### Run the Application
```bash
# Launch the Streamlit interface
streamlit run ui/app.py
```

---

## 📌 Data Sources

- 🧾 [SEC EDGAR](https://www.sec.gov/edgar/searchedgar/companysearch.html)
- 📊 [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`
- 🌎 [FRED API](https://fred.stlouisfed.org/)
- 📁 [Custom CSVs] for portfolios, fund performance, etc.

---

## 💻 Implementation Details

### Agent Architecture
The FinAgent uses LangChain's ReAct agent framework to:
1. Parse user queries for intent
2. Select appropriate tools based on the query type
3. Execute tools and process their outputs
4. Generate comprehensive responses combining multiple data sources

### RAG Implementation
- Uses OpenAI embeddings to index SEC filings
- ChromaDB as the vector store
- Implements semantic search with metadata filtering
- Combines retrieved context with the LLM for accurate answers

### Deployment
This project is designed as a Proof of Concept with Streamlit as the frontend. For production deployment, consider:
- Containerizing with Docker
- Using a production database like PostgreSQL
- Implementing authentication and rate limiting
- Setting up monitoring and logging

---

## 🧠 Future Add-ons

- Chat history memory store (Redis or FAISS)
- Portfolio optimizer tool
- LLM fine-tuning with financial documents
- Integration with Alpaca or Robinhood APIs

---

## 📜 License

MIT License
