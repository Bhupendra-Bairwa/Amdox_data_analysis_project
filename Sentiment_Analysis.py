# src/models/sentiment_analyzer.py
import pandas as pd
import numpy as np
from textblob import TextBlob
import tweepy
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
import snscrape.modules.twitter as sntwitter
from typing import List, Dict

class CryptoSentimentAnalyzer:
    """Analyze sentiment from news and social media"""
    
    def __init__(self, news_api_key: str):
        self.news_api = NewsApiClient(api_key=news_api_key)
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finBERT"
        )
        
    def analyze_text_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of a single text"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Use FinBERT for financial sentiment
        finbert_result = self.sentiment_pipeline(text[:512])[0]
        
        sentiment_label = "NEUTRAL"
        sentiment_score = polarity
        
        if polarity > 0.1:
            sentiment_label = "BULLISH"
        elif polarity < -0.1:
            sentiment_label = "BEARISH"
            
        return {
            'sentiment_label': sentiment_label,
            'sentiment_score': sentiment_score,
            'subjectivity': subjectivity,
            'confidence': finbert_result['score'] if finbert_result else 0.5
        }
    
    def fetch_crypto_news(self, query: str, days_back: int = 7) -> pd.DataFrame:
        """Fetch cryptocurrency news from NewsAPI"""
        try:
            from_date = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            articles = self.news_api.get_everything(
                q=query,
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            
            news_data = []
            for article in articles['articles']:
                sentiment = self.analyze_text_sentiment(article['title'] + " " + 
                                                       (article['description'] or ''))
                news_data.append({
                    'title': article['title'],
                    'description': article['description'],
                    'published_at': article['publishedAt'],
                    'source': article['source']['name'],
                    'url': article['url'],
                    **sentiment
                })
            
            return pd.DataFrame(news_data)
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return pd.DataFrame()
    
    def scrape_twitter_sentiment(self, crypto_symbol: str, tweet_count: int = 100) -> pd.DataFrame:
        """Scrape Twitter data for cryptocurrency sentiment"""
        tweets_data = []
        query = f"${crypto_symbol} OR {crypto_symbol} crypto lang:en"
        
        try:
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                if i >= tweet_count:
                    break
                    
                sentiment = self.analyze_text_sentiment(tweet.content)
                tweets_data.append({
                    'date': tweet.date,
                    'content': tweet.content,
                    'username': tweet.user.username,
                    'like_count': tweet.likeCount,
                    'retweet_count': tweet.retweetCount,
                    **sentiment
                })
                
            return pd.DataFrame(tweets_data)
            
        except Exception as e:
            print(f"Error scraping Twitter: {e}")
            return pd.DataFrame()
    
    def calculate_market_sentiment_index(self, news_df: pd.DataFrame, 
                                        twitter_df: pd.DataFrame) -> float:
        """Calculate overall market sentiment index (0-100)"""
        sentiment_scores = []
        
        # News sentiment contribution (40%)
        if not news_df.empty:
            news_score = (news_df['sentiment_score'] + 1) / 2  # Convert to 0-1
            sentiment_scores.append(news_score.mean() * 0.4)
        
        # Twitter sentiment contribution (60%)
        if not twitter_df.empty:
            # Weight by engagement
            weights = (twitter_df['like_count'] + twitter_df['retweet_count'] * 2)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(twitter_df))
            twitter_score = np.average((twitter_df['sentiment_score'] + 1) / 2, weights=weights)
            sentiment_scores.append(twitter_score * 0.6)
        
        if sentiment_scores:
            sentiment_index = sum(sentiment_scores) * 100
            return min(100, max(0, sentiment_index))
        
        return 50.0  # Neutral sentiment