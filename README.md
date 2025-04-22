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
- 🖥️ **UI Interface**: Optional Streamlit/Gradio frontend for interactive use

---

## 📁 Folder Structure

