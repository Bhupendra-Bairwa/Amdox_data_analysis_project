# src/data_collection/crypto_collector.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from sqlalchemy import create_engine
from typing import List, Dict, Optional
import ccxt  # Cryptocurrency exchange library

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataCollector:
    """Collect real-time and historical cryptocurrency data"""
    
    def __init__(self, db_connection_string: str):
        self.engine = create_engine(db_connection_string)
        self.exchange = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        
    def fetch_historical_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Fetch historical OHLCV data from Binance"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                timeframe='1d', 
                limit=days
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 
                                             'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} days of data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def fetch_real_time_price(self, symbol: str) -> Dict:
        """Fetch current real-time price"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'timestamp': ticker['timestamp'],
                'datetime': ticker['datetime']
            }
        except Exception as e:
            logger.error(f"Error fetching real-time price: {e}")
            return {}
    
    def fetch_market_data_coingecko(self, coin_id: str) -> Dict:
        """Fetch comprehensive market data from CoinGecko"""
        try:
            url = f"{self.coingecko_base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '30',
                'interval': 'daily'
            }
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'prices': data.get('prices', []),
                    'market_caps': data.get('market_caps', []),
                    'total_volumes': data.get('total_volumes', [])
                }
            else:
                logger.error(f"CoinGecko API error: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching CoinGecko data: {e}")
            return {}
    
    def store_to_database(self, df: pd.DataFrame, crypto_id: int):
        """Store collected data to PostgreSQL database"""
        try:
            df['crypto_id'] = crypto_id
            df.reset_index(inplace=True)
            df.to_sql('price_history', self.engine, schema='crypto', 
                     if_exists='append', index=False, method='multi')
            logger.info(f"Stored {len(df)} records to database")
        except Exception as e:
            logger.error(f"Error storing to database: {e}")