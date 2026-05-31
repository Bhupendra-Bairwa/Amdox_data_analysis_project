# src/models/forecasting_models.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class CryptoTradingModels:
    """Advanced forecasting models for cryptocurrency prices"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        
    def arima_forecast(self, data: pd.Series, steps: int = 30) -> Dict:
        """ARIMA model for price forecasting"""
        try:
            # Auto ARIMA with simplified parameters
            model = ARIMA(data, order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            
            # Calculate confidence intervals
            forecast_results = model_fit.get_forecast(steps=steps)
            conf_int = forecast_results.conf_int()
            
            return {
                'forecast': forecast,
                'lower_bound': conf_int.iloc[:, 0],
                'upper_bound': conf_int.iloc[:, 1],
                'model': 'ARIMA'
            }
        except Exception as e:
            print(f"ARIMA error: {e}")
            return {}
    
    def prophet_forecast(self, df: pd.DataFrame, days: int = 30) -> Dict:
        """Facebook Prophet model for time series forecasting"""
        try:
            # Prepare data for Prophet
            prophet_df = df.reset_index()[['timestamp', 'close']]
            prophet_df.columns = ['ds', 'y']
            
            # Create and fit model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.add_country_holidays(country_name='US')
            model.fit(prophet_df)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            return {
                'forecast': forecast['yhat'].tail(days),
                'lower_bound': forecast['yhat_lower'].tail(days),
                'upper_bound': forecast['yhat_upper'].tail(days),
                'model': 'Prophet',
                'components': {
                    'trend': forecast['trend'].tail(days),
                    'weekly': forecast['weekly'].tail(days),
                    'yearly': forecast['yearly'].tail(days)
                }
            }
        except Exception as e:
            print(f"Prophet error: {e}")
            return {}
    
    def prepare_lstm_data(self, data: np.array, lookback: int = 60):
        """Prepare data for LSTM model"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: tuple) -> Sequential:
        """Build LSTM neural network model"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error')
        return model
    
    def lstm_forecast(self, data: pd.Series, lookback: int = 60, 
                     epochs: int = 50, steps: int = 30) -> Dict:
        """LSTM model for price prediction"""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
            
            # Prepare training data
            X, y = self.prepare_lstm_data(scaled_data, lookback)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build and train model
            self.lstm_model = self.build_lstm_model((lookback, 1))
            self.lstm_model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
            
            # Make predictions
            last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
            predictions = []
            
            for _ in range(steps):
                pred = self.lstm_model.predict(last_sequence, verbose=0)[0, 0]
                predictions.append(pred)
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred
            
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            return {
                'forecast': predictions.flatten(),
                'model': 'LSTM'
            }
        except Exception as e:
            print(f"LSTM error: {e}")
            return {}
    
    def ensemble_forecast(self, df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """Combine multiple models for improved accuracy"""
        price_series = df['close']
        
        # Get predictions from different models
        arima_pred = self.arima_forecast(price_series, days)
        prophet_pred = self.prophet_forecast(df, days)
        lstm_pred = self.lstm_forecast(price_series, steps=days)
        
        # Ensemble average with weights
        ensemble_predictions = []
        weights = {'ARIMA': 0.3, 'Prophet': 0.4, 'LSTM': 0.3}
        
        # Align predictions
        min_length = min(
            len(arima_pred.get('forecast', [])),
            len(prophet_pred.get('forecast', [])),
            len(lstm_pred.get('forecast', []))
        )
        
        for i in range(min_length):
            pred_value = 0
            if arima_pred and i < len(arima_pred['forecast']):
                pred_value += arima_pred['forecast'].iloc[i] * weights['ARIMA']
            if prophet_pred and i < len(prophet_pred['forecast']):
                pred_value += prophet_pred['forecast'].iloc[i] * weights['Prophet']
            if lstm_pred and i < len(lstm_pred['forecast']):
                pred_value += lstm_pred['forecast'][i] * weights['LSTM']
            
            ensemble_predictions.append(pred_value)
        
        return pd.DataFrame({
            'date': pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), 
                                 periods=min_length, freq='D'),
            'ensemble_prediction': ensemble_predictions,
            'arima_prediction': arima_pred['forecast'][:min_length] if arima_pred else None,
            'prophet_prediction': prophet_pred['forecast'][:min_length] if prophet_pred else None,
            'lstm_prediction': lstm_pred['forecast'][:min_length] if lstm_pred else None
        })