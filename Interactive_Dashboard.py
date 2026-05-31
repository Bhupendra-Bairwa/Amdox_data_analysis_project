# src/dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
from sqlalchemy import create_engine
import sys
sys.path.append('../')

from data_collection.crypto_collector import CryptoDataCollector
from models.sentiment_analyzer import CryptoSentimentAnalyzer
from models.forecasting_models import CryptoTradingModels

# Page configuration
st.set_page_config(
    page_title="Crypto Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Database connection
@st.cache_resource
def init_db():
    return create_engine('postgresql://user:password@localhost:5432/crypto_analytics')

# Initialize components
@st.cache_resource
def init_analyzers():
    collector = CryptoDataCollector('postgresql://user:password@localhost:5432/crypto_analytics')
    sentiment_analyzer = CryptoSentimentAnalyzer(news_api_key='YOUR_NEWS_API_KEY')
    forecasting_models = CryptoTradingModels()
    return collector, sentiment_analyzer, forecasting_models

def main():
    st.title("📊 Cryptocurrency Analytics Platform")
    st.markdown("### Real-Time Market Analysis & Price Predictions")
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Cryptocurrency selection
    crypto_options = {
        "Bitcoin": "BTC/USDT",
        "Ethereum": "ETH/USDT",
        "Binance Coin": "BNB/USDT",
        "Cardano": "ADA/USDT",
        "Solana": "SOL/USDT",
        "XRP": "XRP/USDT"
    }
    
    selected_crypto = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=list(crypto_options.keys()),
        index=0
    )
    
    symbol = crypto_options[selected_crypto]
    
    # Time period selection
    period = st.sidebar.selectbox(
        "Time Period",
        options=["7D", "30D", "90D", "1Y", "5Y"],
        index=1
    )
    
    # Days mapping
    days_map = {"7D": 7, "30D": 30, "90D": 90, "1Y": 365, "5Y": 1825}
    days = days_map[period]
    
    # Refresh rate
    refresh_rate = st.sidebar.selectbox(
        "Auto Refresh Rate",
        options=["Off", "10 seconds", "30 seconds", "1 minute"],
        index=0
    )
    
    # Initialize components
    collector, sentiment_analyzer, forecasting_models = init_analyzers()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price Analysis", 
        "🔮 Predictions", 
        "💭 Sentiment Analysis",
        "📊 Technical Indicators",
        "⚡ Real-Time Data"
    ])
    
    # Tab 1: Price Analysis
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        # Fetch real-time data
        real_time_data = collector.fetch_real_time_price(symbol)
        
        if real_time_data:
            with col1:
                st.metric(
                    label="Current Price",
                    value=f"${real_time_data.get('price', 0):,.2f}",
                    delta=f"24h: +2.5%"
                )
            
            with col2:
                st.metric(
                    label="24h Volume",
                    value=f"${real_time_data.get('volume', 0):,.0f}"
                )
            
            with col3:
                st.metric(
                    label="24h High/Low",
                    value=f"${real_time_data.get('high', 0):,.0f}",
                    delta=f"Low: ${real_time_data.get('low', 0):,.0f}"
                )
        
        # Fetch historical data
        df = collector.fetch_historical_data(symbol, days)
        
        if not df.empty:
            # Candlestick chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{selected_crypto} Price', 'Volume'),
                row_width=[0.2, 0.7]
            )
            
            # Candlestick trace
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Volume trace
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(df['close'], df['open'])]
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f'{selected_crypto} Price Chart',
                yaxis_title='Price (USD)',
                xaxis_title='Date',
                height=600,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key statistics
            st.subheader("📊 Key Statistics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Average Price", f"${df['close'].mean():,.0f}")
            with col2:
                st.metric("Max Price", f"${df['close'].max():,.0f}")
            with col3:
                st.metric("Min Price", f"${df['close'].min():,.0f}")
            with col4:
                st.metric("Volatility", f"{df['close'].pct_change().std()*100:.2f}%")
            with col5:
                st.metric("Volume (Avg)", f"${df['volume'].mean():,.0f}")
    
    # Tab 2: Predictions
    with tab2:
        st.subheader("🔮 Price Predictions")
        
        prediction_days = st.slider("Prediction Horizon (days)", 7, 90, 30)
        
        if st.button("Generate Predictions", type="primary"):
            with st.spinner("Generating predictions using ensemble model..."):
                # Generate predictions
                predictions = forecasting_models.ensemble_forecast(df, prediction_days)
                
                if not predictions.empty:
                    # Plot predictions
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=df.index[-90:], 
                        y=df['close'][-90:],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Ensemble predictions
                    fig.add_trace(go.Scatter(
                        x=predictions['date'],
                        y=predictions['ensemble_prediction'],
                        mode='lines+markers',
                        name='Ensemble Prediction',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    # Individual model predictions
                    fig.add_trace(go.Scatter(
                        x=predictions['date'],
                        y=predictions['arima_prediction'],
                        mode='lines',
                        name='ARIMA',
                        line=dict(color='green', width=1, dash='dot')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=predictions['date'],
                        y=predictions['prophet_prediction'],
                        mode='lines',
                        name='Prophet',
                        line=dict(color='orange', width=1, dash='dot')
                    ))
                    
                    fig.update_layout(
                        title='Price Predictions - Ensemble Model',
                        xaxis_title='Date',
                        yaxis_title='Price (USD)',
                        height=500,
                        template='plotly_dark',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction metrics
                    st.subheader("Prediction Insights")
                    col1, col2, col3 = st.columns(3)
                    
                    last_price = df['close'].iloc[-1]
                    future_price = predictions['ensemble_prediction'].iloc[-1]
                    price_change = ((future_price - last_price) / last_price) * 100
                    
                    with col1:
                        st.metric(
                            "Expected Price",
                            f"${future_price:,.2f}",
                            delta=f"{price_change:.1f}%"
                        )
                    
                    with col2:
                        max_price = predictions['ensemble_prediction'].max()
                        max_date = predictions['date'][predictions['ensemble_prediction'].idxmax()]
                        st.metric(
                            "Maximum Expected",
                            f"${max_price:,.2f}",
                            delta=f"on {max_date.strftime('%Y-%m-%d')}"
                        )
                    
                    with col3:
                        min_price = predictions['ensemble_prediction'].min()
                        st.metric(
                            "Minimum Expected",
                            f"${min_price:,.2f}"
                        )
                    
                    # Download predictions
                    csv = predictions.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv,
                        file_name=f"{selected_crypto}_predictions.csv",
                        mime="text/csv"
                    )
    
    # Tab 3: Sentiment Analysis
    with tab3:
        st.subheader("💭 Market Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Fetching latest news and social media sentiment...")
            
            # Fetch news
            news_df = sentiment_analyzer.fetch_crypto_news(
                selected_crypto.lower(), days_back=7
            )
            
            if not news_df.empty:
                st.subheader("📰 Latest News")
                for idx, row in news_df.head(5).iterrows():
                    with st.expander(f"{row['title']}"):
                        st.write(f"**Source:** {row['source']}")
                        st.write(f"**Sentiment:** {row['sentiment_label']} (Score: {row['sentiment_score']:.2f})")
                        st.write(f"**Description:** {row['description']}")
                        st.markdown(f"[Read more]({row['url']})")
        
        with col2:
            st.subheader("Sentiment Metrics")
            
            # Calculate sentiment index
            twitter_df = sentiment_analyzer.scrape_twitter_sentiment(
                selected_crypto.replace(" ", ""), 50
            )
            
            sentiment_index = sentiment_analyzer.calculate_market_sentiment_index(
                news_df, twitter_df
            )
            
            # Gauge chart for sentiment
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = sentiment_index,
                title = {'text': "Market Sentiment Index"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "gray"},
                        {'range': [66, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment breakdown
            if not twitter_df.empty:
                sentiment_counts = twitter_df['sentiment_label'].value_counts()
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Twitter Sentiment Distribution",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    
    # Tab 4: Technical Indicators
    with tab4:
        st.subheader("📊 Technical Analysis Indicators")
        
        # Calculate indicators
        df_ta = df.copy()
        
        # Moving averages
        df_ta['SMA_20'] = df_ta['close'].rolling(window=20).mean()
        df_ta['SMA_50'] = df_ta['close'].rolling(window=50).mean()
        df_ta['EMA_12'] = df_ta['close'].ewm(span=12, adjust=False).mean()
        
        # RSI
        delta = df_ta['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_ta['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df_ta['BB_Middle'] = df_ta['close'].rolling(window=20).mean()
        bb_std = df_ta['close'].rolling(window=20).std()
        df_ta['BB_Upper'] = df_ta['BB_Middle'] + (bb_std * 2)
        df_ta['BB_Lower'] = df_ta['BB_Middle'] - (bb_std * 2)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price with Indicators', 'RSI', 'Bollinger Bands'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price and moving averages
        fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['close'], name='Price', line=dict(color='white')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['SMA_20'], name='SMA 20', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['SMA_50'], name='SMA 50', line=dict(color='orange')), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Middle'], name='BB Middle', line=dict(color='yellow')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=3, col=1)
        
        fig.update_layout(height=800, template='plotly_dark', showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Signal generation
        current_rsi = df_ta['RSI'].iloc[-1]
        current_price = df_ta['close'].iloc[-1]
        current_sma_20 = df_ta['SMA_20'].iloc[-1]
        
        st.subheader("📈 Trading Signals")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if current_rsi > 70:
                st.warning("🟡 OVERBOUGHT - Consider selling")
            elif current_rsi < 30:
                st.success("🟢 OVERSOLD - Consider buying")
            else:
                st.info("⚪ NEUTRAL - No strong signal")
        
        with col2:
            if current_price > current_sma_20:
                st.success("🟢 BULLISH - Price above SMA20")
            else:
                st.warning("🔴 BEARISH - Price below SMA20")
        
        with col3:
            st.metric("Current RSI", f"{current_rsi:.1f}")
    
    # Tab 5: Real-Time Data
    with tab5:
        st.subheader("⚡ Real-Time Market Data")
        
        # Auto-refresh logic
        if refresh_rate != "Off":
            refresh_seconds = {"Off": 0, "10 seconds": 10, "30 seconds": 30, "1 minute": 60}[refresh_rate]
            st.empty()
            time_placeholder = st.empty()
        
        # Real-time price chart
        real_time_chart = st.line_chart()
        
        # WebSocket simulation (simplified)
        if st.button("Start Real-Time Stream"):
            status_placeholder = st.info("Streaming real-time data...")
            
            # Simulate real-time updates
            for i in range(50):  # 50 updates
                rt_data = collector.fetch_real_time_price(symbol)
                if rt_data:
                    # Update display
                    status_placeholder.info(f"Latest Price: ${rt_data['price']:,.2f} | Time: {rt_data['datetime']}")
                    
                    # Update chart (simplified)
                    # In production, use proper streaming
                
                time.sleep(1)
        
        # Order book simulation
        st.subheader("Order Book Depth")
        order_book_data = pd.DataFrame({
            'Price': [50000, 49900, 49800, 49700, 49600],
            'Bid Size': [2.5, 5.2, 8.1, 12.3, 15.7],
            'Ask Size': [3.1, 6.4, 9.2, 11.8, 14.2]
        })
        st.dataframe(order_book_data, use_container_width=True)

if __name__ == "__main__":
    main()