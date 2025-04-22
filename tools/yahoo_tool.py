from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class YahooInput(BaseModel):
    """Input for Yahoo Finance tool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL', 'MSFT')")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    info_type: str = Field("price", description="Type of information to retrieve: 'price', 'info', 'financials', 'actions', 'dividends'")

class YahooTool(BaseTool):
    name = "yahoo_tool"
    description = """
    Retrieve stock market data from Yahoo Finance.
    Use this to get historical price data, company information, financial statements, 
    dividends, and stock splits.
    
    Types of information available:
    - price: Historical stock prices
    - info: General company information and key statistics
    - financials: Income statement, balance sheet, cash flow details
    - actions: Stock splits and dividends
    - dividends: Historical dividend payments
    """
    args_schema: Type[BaseModel] = YahooInput
    
    def _run(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None, info_type: str = "price") -> str:
        try:
            # Default to 1 year of data if no date range specified
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            # Get ticker data
            stock = yf.Ticker(ticker)
            
            # Process based on info_type
            if info_type.lower() == "price":
                data = stock.history(start=start_date, end=end_date)
                if data.empty:
                    return f"No price data found for {ticker} in the specified date range."
                
                result = f"Historical prices for {ticker} from {start_date} to {end_date}:\n"
                result += data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).to_string()
                result += f"\n\nSummary Statistics (Close Price):\n"
                result += f"Mean: ${data['Close'].mean():.2f}\n"
                result += f"Min: ${data['Close'].min():.2f}\n"
                result += f"Max: ${data['Close'].max():.2f}\n"
                result += f"Latest: ${data['Close'].iloc[-1]:.2f}\n"
                return result
                
            elif info_type.lower() == "info":
                info = stock.info
                result = f"Company Information for {ticker} ({info.get('longName', ticker)}):\n\n"
                result += f"Sector: {info.get('sector', 'N/A')}\n"
                result += f"Industry: {info.get('industry', 'N/A')}\n"
                result += f"Market Cap: ${info.get('marketCap', 0) / 1e9:.2f}B\n"
                result += f"P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
                result += f"EPS: ${info.get('trailingEps', 'N/A')}\n"
                result += f"52-Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
                result += f"52-Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}\n"
                result += f"Dividend Yield: {info.get('dividendYield', 0) * 100:.2f}%\n"
                result += f"Beta: {info.get('beta', 'N/A')}\n"
                result += f"\nBusiness Summary: {info.get('longBusinessSummary', 'N/A')[:500]}...\n"
                return result
                
            elif info_type.lower() == "financials":
                income_stmt = stock.income_stmt
                balance_sheet = stock.balance_sheet
                cash_flow = stock.cashflow
                
                result = f"Financial Statement Highlights for {ticker}:\n\n"
                
                # Income Statement
                result += "Income Statement (Recent Year):\n"
                if not income_stmt.empty:
                    recent_income = income_stmt.iloc[:, 0]
                    result += f"Total Revenue: ${recent_income.get('Total Revenue', 0) / 1e9:.2f}B\n"
                    result += f"Gross Profit: ${recent_income.get('Gross Profit', 0) / 1e9:.2f}B\n"
                    result += f"Net Income: ${recent_income.get('Net Income', 0) / 1e9:.2f}B\n"
                
                # Balance Sheet
                result += "\nBalance Sheet (Recent Quarter):\n"
                if not balance_sheet.empty:
                    recent_bs = balance_sheet.iloc[:, 0]
                    result += f"Total Assets: ${recent_bs.get('Total Assets', 0) / 1e9:.2f}B\n"
                    result += f"Total Liabilities: ${recent_bs.get('Total Liabilities Net Minority Interest', 0) / 1e9:.2f}B\n"
                    result += f"Total Equity: ${recent_bs.get('Total Stockholder Equity', 0) / 1e9:.2f}B\n"
                
                # Cash Flow
                result += "\nCash Flow (Recent Year):\n"
                if not cash_flow.empty:
                    recent_cf = cash_flow.iloc[:, 0]
                    result += f"Operating Cash Flow: ${recent_cf.get('Operating Cash Flow', 0) / 1e9:.2f}B\n"
                    result += f"Free Cash Flow: ${recent_cf.get('Free Cash Flow', 0) / 1e9:.2f}B\n"
                
                return result
                
            elif info_type.lower() == "actions":
                actions = stock.actions
                if actions.empty:
                    return f"No stock split or dividend actions found for {ticker} in the specified date range."
                result = f"Stock Actions for {ticker}:\n\n"
                result += actions.to_string()
                return result
                
            elif info_type.lower() == "dividends":
                dividends = stock.dividends
                if dividends.empty:
                    return f"No dividend data found for {ticker}."
                result = f"Dividend History for {ticker}:\n\n"
                result += dividends.to_string()
                return result
                
            else:
                return f"Invalid info_type: {info_type}. Use 'price', 'info', 'financials', 'actions', or 'dividends'."
                
        except Exception as e:
            return f"Error retrieving Yahoo Finance data: {str(e)}" 