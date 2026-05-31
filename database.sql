-- database_schema.sql
-- Create database
CREATE DATABASE crypto_analytics;

-- Connect to database
\c crypto_analytics;

-- Create schema
CREATE SCHEMA IF NOT EXISTS crypto;

-- Table for cryptocurrency metadata
CREATE TABLE crypto.cryptocurrencies (
    crypto_id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(50) NOT NULL,
    coin_gecko_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for historical price data
CREATE TABLE crypto.price_history (
    price_id SERIAL PRIMARY KEY,
    crypto_id INTEGER REFERENCES crypto.cryptocurrencies(crypto_id),
    timestamp TIMESTAMP NOT NULL,
    open_price DECIMAL(20, 8),
    high_price DECIMAL(20, 8),
    low_price DECIMAL(20, 8),
    close_price DECIMAL(20, 8),
    volume DECIMAL(30, 8),
    market_cap DECIMAL(30, 2),
    UNIQUE(crypto_id, timestamp)
);

-- Table for sentiment data
CREATE TABLE crypto.sentiment_data (
    sentiment_id SERIAL PRIMARY KEY,
    crypto_id INTEGER REFERENCES crypto.cryptocurrencies(crypto_id),
    timestamp TIMESTAMP,
    source VARCHAR(50),
    sentiment_score DECIMAL(5, 4),
    positive_count INTEGER,
    negative_count INTEGER,
    neutral_count INTEGER,
    confidence DECIMAL(5, 4)
);

-- Table for predictions
CREATE TABLE crypto.predictions (
    prediction_id SERIAL PRIMARY KEY,
    crypto_id INTEGER REFERENCES crypto.cryptocurrencies(crypto_id),
    prediction_date TIMESTAMP,
    predicted_price DECIMAL(20, 8),
    lower_bound DECIMAL(20, 8),
    upper_bound DECIMAL(20, 8),
    model_used VARCHAR(50),
    confidence_score DECIMAL(5, 4)
);

-- Create indexes for performance
CREATE INDEX idx_price_timestamp ON crypto.price_history(timestamp);
CREATE INDEX idx_price_crypto_time ON crypto.price_history(crypto_id, timestamp);
CREATE INDEX idx_sentiment_time ON crypto.sentiment_data(crypto_id, timestamp);