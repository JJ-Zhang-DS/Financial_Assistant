from langchain.tools import BaseTool
from typing import Optional, Type, List
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import os
import plotly.express as px
import json

class PlotInput(BaseModel):
    """Input for plotting tool."""
    data_source: str = Field(..., description="Data to plot: 'fred', 'yahoo', or JSON data directly")
    data_id: Optional[str] = Field(None, description="ID for predefined data (e.g., 'CPIAUCSL' for FRED, 'AAPL' for Yahoo)")
    chart_type: str = Field("line", description="Type of chart: 'line', 'bar', 'scatter', 'histogram', 'pie'")
    title: Optional[str] = Field(None, description="Title for the chart")
    x_label: Optional[str] = Field(None, description="Label for x-axis")
    y_label: Optional[str] = Field(None, description="Label for y-axis")
    data_json: Optional[str] = Field(None, description="JSON data to plot directly")

class PlotTool(BaseTool):
    name = "plot_tool"
    description = """
    Generate visualizations of financial data.
    Use this to create line charts, bar charts, scatter plots, histograms, and pie charts 
    from FRED macroeconomic data, Yahoo Finance stock data, or custom JSON data.
    
    For predefined data sources:
    - Use data_source='fred' and data_id='SERIES_ID' for FRED data
    - Use data_source='yahoo' and data_id='TICKER' for Yahoo Finance data
    
    For custom data:
    - Use data_source='json' and provide data_json as a stringified JSON object
    """
    args_schema: Type[BaseModel] = PlotInput
    
    def __init__(self):
        super().__init__()
        self.output_dir = "data/plots"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _run(
        self, 
        data_source: str,
        data_id: Optional[str] = None,
        chart_type: str = "line",
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        data_json: Optional[str] = None
    ) -> str:
        try:
            # Get data based on source
            df = self._get_data(data_source, data_id, data_json)
            if isinstance(df, str):  # Error message
                return df
            
            # Generate the plot
            if chart_type.lower() in ["line", "bar", "scatter", "histogram", "pie"]:
                plot_path = self._create_plot(df, chart_type, title, x_label, y_label)
                return f"Plot generated successfully and saved to {plot_path}"
            else:
                return f"Unsupported chart type: {chart_type}. Supported types: line, bar, scatter, histogram, pie."
            
        except Exception as e:
            return f"Error generating plot: {str(e)}"
    
    def _get_data(self, data_source: str, data_id: Optional[str], data_json: Optional[str]) -> pd.DataFrame:
        """Get data from specified source."""
        if data_source.lower() == "fred":
            if not data_id:
                return "For FRED data, you must provide a data_id (series ID)"
            # This is a simplified implementation. In a real tool, you'd use FREDTool
            # to fetch real-time data. Here we'll use dummy data.
            return pd.DataFrame({
                'date': pd.date_range(start='2020-01-01', periods=24, freq='M'),
                'value': [100 + i + i*i*0.01 for i in range(24)]
            }).set_index('date')
            
        elif data_source.lower() == "yahoo":
            if not data_id:
                return "For Yahoo data, you must provide a data_id (ticker symbol)"
            # This is a simplified implementation. In a real tool, you'd use YahooTool
            # to fetch real-time data. Here we'll use dummy data.
            return pd.DataFrame({
                'date': pd.date_range(start='2020-01-01', periods=24, freq='M'),
                'close': [100 + i + i*i*0.05 for i in range(24)]
            }).set_index('date')
            
        elif data_source.lower() == "json":
            if not data_json:
                return "For JSON data, you must provide data_json"
            try:
                data = json.loads(data_json)
                return pd.DataFrame(data)
            except json.JSONDecodeError:
                return f"Invalid JSON format: {data_json}"
                
        else:
            return f"Unsupported data source: {data_source}. Use 'fred', 'yahoo', or 'json'."
    
    def _create_plot(self, df: pd.DataFrame, chart_type: str, title: Optional[str], x_label: Optional[str], y_label: Optional[str]) -> str:
        """Create plot based on data and chart type."""
        plt.figure(figsize=(10, 6))
        
        if chart_type.lower() == "line":
            df.plot(kind='line')
        elif chart_type.lower() == "bar":
            df.plot(kind='bar')
        elif chart_type.lower() == "scatter":
            if len(df.columns) >= 2:
                plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
            else:
                plt.scatter(df.index, df.iloc[:, 0])
        elif chart_type.lower() == "histogram":
            df.plot(kind='hist')
        elif chart_type.lower() == "pie":
            df.iloc[-1].plot(kind='pie')
        
        if title:
            plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.output_dir, f"{chart_type}_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path 