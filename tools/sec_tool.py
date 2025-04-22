from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import os
import re
import pandas as pd
from bs4 import BeautifulSoup
import requests
from pypdf import PdfReader
import tempfile

class SECInput(BaseModel):
    """Input for SEC filing parser tool."""
    ticker: str = Field(..., description="Company ticker symbol (e.g., 'AAPL', 'MSFT')")
    filing_type: str = Field("10-K", description="SEC filing type (e.g., '10-K', '10-Q', '8-K')")
    section: Optional[str] = Field(None, description="Section to extract (e.g., 'Risk Factors', 'MD&A', 'Financial Statements')")
    filing_path: Optional[str] = Field(None, description="Path to local filing if available")

class SECParserTool(BaseTool):
    name = "sec_parser_tool"
    description = """
    Parse and extract information from SEC filings like 10-K and 10-Q reports.
    Can search for specific sections or tables in the filings.
    Use this to find risk factors, management discussion, financial statements, 
    and other regulatory information from company filings.
    """
    args_schema: Type[BaseModel] = SECInput
    
    def __init__(self):
        super().__init__()
        self.data_dir = "data/sec_filings"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _run(self, ticker: str, filing_type: str = "10-K", section: Optional[str] = None, filing_path: Optional[str] = None) -> str:
        try:
            # If a specific filing path is provided, use that
            if filing_path and os.path.exists(filing_path):
                return self._parse_filing(filing_path, section)
            
            # Otherwise look in our data directory for filings
            ticker = ticker.upper()
            filing_type = filing_type.upper()
            
            # Check if we have the filing locally
            ticker_dir = os.path.join(self.data_dir, ticker)
            if os.path.exists(ticker_dir):
                for filename in os.listdir(ticker_dir):
                    if filing_type in filename:
                        filing_path = os.path.join(ticker_dir, filename)
                        return self._parse_filing(filing_path, section)
            
            # If we don't have the filing, inform the user
            return f"SEC filing {filing_type} for {ticker} not found in local storage. Please provide a filing_path or add the filing to {ticker_dir}."
            
        except Exception as e:
            return f"Error parsing SEC filing: {str(e)}"
    
    def _parse_filing(self, filing_path: str, section: Optional[str] = None) -> str:
        """Parse an SEC filing from a local file."""
        file_ext = os.path.splitext(filing_path)[1].lower()
        
        # For PDF files
        if file_ext == '.pdf':
            return self._parse_pdf(filing_path, section)
        
        # For HTML files
        elif file_ext in ['.html', '.htm']:
            return self._parse_html(filing_path, section)
        
        # For text files
        elif file_ext == '.txt':
            return self._parse_text(filing_path, section)
        
        else:
            return f"Unsupported file format: {file_ext}. Only PDF, HTML, and TXT are supported."
    
    def _parse_pdf(self, pdf_path: str, section: Optional[str] = None) -> str:
        """Extract content from a PDF filing."""
        try:
            pdf = PdfReader(pdf_path)
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            
            if section:
                # Simple approach: find section headers and extract content until next section
                section_pattern = re.compile(f"{re.escape(section)}.*?(?=Item \d|$)", re.DOTALL | re.IGNORECASE)
                matches = section_pattern.findall(text)
                if matches:
                    return f"Section '{section}' from {os.path.basename(pdf_path)}:\n\n{matches[0][:2000]}...\n\n(Content truncated for readability)"
                else:
                    return f"Section '{section}' not found in {os.path.basename(pdf_path)}"
            else:
                # Return a summary of the filing
                return f"Summary of {os.path.basename(pdf_path)}:\n\nTotal pages: {len(pdf.pages)}\nFile size: {os.path.getsize(pdf_path) / 1024 / 1024:.2f} MB\n\nExcerpt:\n{text[:2000]}...\n\n(Content truncated for readability)"
        
        except Exception as e:
            return f"Error parsing PDF: {str(e)}"
    
    def _parse_html(self, html_path: str, section: Optional[str] = None) -> str:
        """Extract content from an HTML filing."""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html = f.read()
            
            soup = BeautifulSoup(html, 'lxml')
            # Remove scripts, styles and comments
            for script in soup(["script", "style"]):
                script.extract()
            
            text = soup.get_text(separator="\n")
            
            if section:
                # Simple approach: find section headers and extract content until next section
                section_pattern = re.compile(f"{re.escape(section)}.*?(?=Item \d|$)", re.DOTALL | re.IGNORECASE)
                matches = section_pattern.findall(text)
                if matches:
                    return f"Section '{section}' from {os.path.basename(html_path)}:\n\n{matches[0][:2000]}...\n\n(Content truncated for readability)"
                else:
                    return f"Section '{section}' not found in {os.path.basename(html_path)}"
            else:
                # Return a summary of the filing
                return f"Summary of {os.path.basename(html_path)}:\n\nFile size: {os.path.getsize(html_path) / 1024 / 1024:.2f} MB\n\nExcerpt:\n{text[:2000]}...\n\n(Content truncated for readability)"
        
        except Exception as e:
            return f"Error parsing HTML: {str(e)}"
    
    def _parse_text(self, text_path: str, section: Optional[str] = None) -> str:
        """Extract content from a text filing."""
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if section:
                # Simple approach: find section headers and extract content until next section
                section_pattern = re.compile(f"{re.escape(section)}.*?(?=Item \d|$)", re.DOTALL | re.IGNORECASE)
                matches = section_pattern.findall(text)
                if matches:
                    return f"Section '{section}' from {os.path.basename(text_path)}:\n\n{matches[0][:2000]}...\n\n(Content truncated for readability)"
                else:
                    return f"Section '{section}' not found in {os.path.basename(text_path)}"
            else:
                # Return a summary of the filing
                return f"Summary of {os.path.basename(text_path)}:\n\nFile size: {os.path.getsize(text_path) / 1024 / 1024:.2f} MB\n\nExcerpt:\n{text[:2000]}...\n\n(Content truncated for readability)"
        
        except Exception as e:
            return f"Error parsing text file: {str(e)}" 