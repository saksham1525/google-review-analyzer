"""Sentiment analysis using DistilBERT"""

from transformers import pipeline


class SentimentAnalyzer:
    """Analyzes review sentiment"""
    
    def __init__(self):
        print("Loading sentiment model...")
        self.analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
        print("Model loaded!")
    
    def analyze(self, text):
        """Returns sentiment label and confidence score"""
        if not text or len(text.strip()) == 0:
            return {'label': 'NEUTRAL', 'score': 0.0}
        return self.analyzer(text[:512])[0]
    
    def analyze_reviews(self, df):
        """Add sentiment columns to dataframe"""
        print(f"Analyzing {len(df)} reviews...")
        
        sentiments = [
            self.analyze(row['caption']) if row['has_text'] else {'label': 'NEUTRAL', 'score': 0.0}
            for _, row in df.iterrows()
        ]
        
        # Add to dataframe
        df['sentiment'] = [s['label'] for s in sentiments]
        df['sentiment_score'] = [s['score'] for s in sentiments]
        
        counts = df[df['has_text']]['sentiment'].value_counts()
        print(f"Sentiment: Positive={counts.get('POSITIVE', 0)} Negative={counts.get('NEGATIVE', 0)} Neutral={counts.get('NEUTRAL', 0)}")
        return df
