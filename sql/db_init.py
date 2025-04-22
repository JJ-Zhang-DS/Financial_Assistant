import sqlite3
import pandas as pd
import os
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import datetime
import time

load_dotenv()

class FinanceDBInitializer:
    def __init__(self, db_path: str = "sql/db.sqlite3"):
        """Initialize the finance database."""
        self.db_path = db_path
        self.db_dir = os.path.dirname(db_path)
        
        # Create database directory if it doesn't exist
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Initialize FRED API if available
        fred_api_key = os.getenv("FRED_API_KEY")
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
    
    def create_tables(self):
        """Create the database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create stock_prices table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            ticker TEXT,
            date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adjusted_close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
        ''')
        
        # Create stock_fundamentals table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_fundamentals (
            ticker TEXT,
            date DATE,
            revenue REAL,
            revenue REAL,
            PRIMARY KEY (ticker, date)
        )
        ''') 