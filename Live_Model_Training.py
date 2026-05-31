# src/models/live_trainer.py
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from tensorflow.keras.models import load_model
import redis
from typing import Dict, Any

class LiveModelTrainer:
    """Train and update models with live streaming data"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.model_cache = {}
        self.buffer_size = 1000
        self.data_buffer = []
        
    async def stream_websocket_data(self, symbol: str):
        """Stream real-time data from WebSocket"""
        uri = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"
        
        async with websockets.connect(uri) as websocket:
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    trade_data = {
                        'timestamp': datetime.fromtimestamp(data['T'] / 1000),
                        'price': float(data['p']),
                        'volume': float(data['q']),
                        'is_buyer_maker': data['m']
                    }
                    
                    self.process_live_data(trade_data)
                    
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    break
    
    def process_live_data(self, trade_data: Dict):
        """Process and store live trading data"""
        self.data_buffer.append(trade_data)
        
        # Keep only last N items
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
        
        # Store in Redis for real-time access
        self.redis_client.lpush('live_trades', json.dumps(trade_data))
        self.redis_client.ltrim('live_trades', 0, self.buffer_size)
        
        # Update models every 100 trades
        if len(self.data_buffer) % 100 == 0:
            self.update_models()
    
    def update_models(self):
        """Update forecasting models with new data"""
        df = pd.DataFrame(self.data_buffer)
        
        # Update statistical models
        self.update_arima_model(df)
        self.update_prophet_model(df)
        
        # Incremental LSTM training
        if len(self.data_buffer) > 200:
            self.update_lstm_incremental(df)
    
    def update_arima_model(self, df: pd.DataFrame):
        """Update ARIMA model incrementally"""
        # Implementation for incremental ARIMA update
        pass
    
    def update_prophet_model(self, df: pd.DataFrame):
        """Update Prophet model with new data"""
        # Implementation for Prophet model update
        pass
    
    def update_lstm_incremental(self, df: pd.DataFrame):
        """Incremental training for LSTM model"""
        # Implementation for online LSTM learning
        pass
    
    def get_live_predictions(self) -> Dict[str, Any]:
        """Get real-time predictions from all models"""
        if len(self.data_buffer) < 100:
            return {'status': 'insufficient_data'}
        
        df = pd.DataFrame(self.data_buffer)
        current_price = df['price'].iloc[-1]
        
        # Calculate short-term predictions
        predictions = {
            'current_price': current_price,
            'short_term_trend': self.calculate_short_term_trend(df),
            'volatility_estimate': df['price'].pct_change().std(),
            'momentum_score': self.calculate_momentum(df),
            'timestamp': datetime.now().isoformat()
        }
        
        return predictions
    
    def calculate_short_term_trend(self, df: pd.DataFrame, window: int = 20) -> str:
        """Calculate short-term price trend"""
        if len(df) < window:
            return "insufficient_data"
        
        recent_prices = df['price'].tail(window)
        slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        
        if slope > 0.01:
            return "strong_bullish"
        elif slope > 0:
            return "mild_bullish"
        elif slope > -0.01:
            return "neutral"
        elif slope > -0.02:
            return "mild_bearish"
        else:
            return "strong_bearish"
    
    def calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate momentum score (-100 to 100)"""
        if len(df) < 50:
            return 0
        
        # RSI calculation
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD calculation
        exp1 = df['price'].ewm(span=12, adjust=False).mean()
        exp2 = df['price'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = macd - signal
        
        # Combine indicators
        momentum_score = (rsi.iloc[-1] - 50) * 0.6 + (macd_histogram.iloc[-1] * 100) * 0.4
        
        return np.clip(momentum_score, -100, 100)