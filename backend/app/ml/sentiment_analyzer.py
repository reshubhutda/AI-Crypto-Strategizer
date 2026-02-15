import requests
from app.core.config import settings
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        # Load sentiment model (FinBERT - trained on financial news)
        print("Loading sentiment analysis model...")
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
        print("Model loaded successfully!")
    
    def fetch_news(self, crypto_symbol: str, limit: int = 10):
        "Fetch latest crypto news from NewsData.io"
        coin_code = crypto_symbol[:3]  # BTC, ETH, SOL
    
        # Search with both the code and "cryptocurrency" to get relevant results
        search_query = f"{coin_code} cryptocurrency"
        url = "https://newsdata.io/api/1/news"
        params = {
            "apikey": settings.NEWSDATA_API_KEY,
            "q": search_query,
            "language": "en",
            "category": "business,technology"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract headlines
            articles = data.get("results", [])
            headlines = [article.get("title", "") for article in articles[:limit]]
            
            return headlines
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def analyze_sentiment(self, text: str):
        "Analyze sentiment of a single text"
        try:
            result = self.sentiment_model(text[:512])[0]  # Limit to 512 chars
            
            # FinBERT returns: positive, negative, neutral
            label = result['label'].lower()
            score = result['score']
            
            # Convert to numerical score (-1 to +1)
            if label == 'positive':
                sentiment_score = score
            elif label == 'negative':
                sentiment_score = -score
            else:  # neutral
                sentiment_score = 0
            
            return {
                "label": label,
                "score": score,
                "sentiment_score": sentiment_score
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"label": "neutral", "score": 0.5, "sentiment_score": 0}
    
    def get_overall_sentiment(self, crypto_symbol: str, article_limit: int = 10):
        "Get overall sentiment for a cryptocurrency"
        
        # Fetch news
        headlines = self.fetch_news(crypto_symbol, limit=article_limit)
        
        if not headlines:
            return {
                "signal": "NEUTRAL",
                "sentiment_score": 0,
                "confidence": 0,
                "articles_analyzed": 0,
                "headlines": []
            }
        
        # Analyze each headline
        sentiments = []
        analyzed_headlines = []
        
        for headline in headlines:
            sentiment = self.analyze_sentiment(headline)
            sentiments.append(sentiment['sentiment_score'])
            analyzed_headlines.append({
                "headline": headline,
                "sentiment": sentiment['label'],
                "score": sentiment['score']
            })
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Determine signal
        if avg_sentiment > 0.2:
            signal = "BULLISH"
        elif avg_sentiment < -0.2:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
        
        return {
            "signal": signal,
            "sentiment_score": avg_sentiment,
            "confidence": abs(avg_sentiment),
            "articles_analyzed": len(headlines),
            "headlines": analyzed_headlines[:5]  # Return top 5
        }

sentiment_analyzer = SentimentAnalyzer()