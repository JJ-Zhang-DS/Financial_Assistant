from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import os
from fredapi import Fred
from dotenv import load_dotenv
import pandas as pd
import datetime

load_dotenv()

class FREDInput(BaseModel):
    """Input for FRED API tool."""
    series_id: str = Field(..., description="FRED series ID (e.g., 'GDP', 'CPIAUCSL', 'FEDFUNDS')")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")

class FREDTool(BaseTool):
    name = "fred_tool"
    description = """
    Retrieve macroeconomic data from the Federal Reserve Economic Data (FRED).
    Use this to get time series data for indicators like GDP, inflation (CPI), 
    interest rates, unemployment, and other economic metrics.
    
    Common series IDs:
    - GDP: Gross Domestic Product
    - CPIAUCSL: Consumer Price Index (inflation)
    - FEDFUNDS: Federal Funds Rate (interest rate)
    - UNRATE: Unemployment Rate
    - M2: Money Supply
    """
    args_schema: Type[BaseModel] = FREDInput
    
    def __init__(self):
        super().__init__()
        fred_api_key = os.getenv("FRED_API_KEY")
        if not fred_api_key:
            raise ValueError("FRED_API_KEY not found in environment variables")
        self.fred = Fred(api_key=fred_api_key)
    
    def _run(self, series_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        try:
            # Default to 5 years of data if no date range specified
            if not end_date:
                end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.datetime.now() - datetime.timedelta(days=5*365)).strftime("%Y-%m-%d")
                
            # Get data from FRED
            data = self.fred.get_series(
                series_id, 
                observation_start=start_date,
                observation_end=end_date
            )
            
            # Get series information
            series_info = self.fred.get_series_info(series_id)
            title = series_info['title']
            
            # Format the result
            if len(data) == 0:
                return f"No data found for series {series_id} in the specified date range."
            
            result = f"FRED Data for {title} ({series_id}):\n"
            result += data.to_string()
            
            # Add some basic statistics
            result += f"\n\nSummary Statistics:\n"
            result += f"Mean: {data.mean():.2f}\n"
            result += f"Min: {data.min():.2f}\n"
            result += f"Max: {data.max():.2f}\n"
            result += f"Latest value ({data.index[-1]}): {data.iloc[-1]:.2f}\n"
            
            return result
        
        except Exception as e:
            return f"Error retrieving FRED data: {str(e)}" 