"""Sentiment analysis using DistilBERT"""

from transformers import pipeline


class SentimentAnalyzer:
    """Analyzes review sentiment"""
    
    def __init__(self):
        print("Loading sentiment model...")
        # Using a simple sentiment model
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        print("Model loaded!")
    
    def analyze(self, text):
        """Returns sentiment label and confidence score"""
        if not text or len(text.strip()) == 0:
            return {'label': 'NEUTRAL', 'score': 0.0}
        
        result = self.analyzer(text[:512])[0]  # Truncate to 512 chars - model limit
        return result
    
    def analyze_reviews(self, df):
        """Add sentiment and sentiment_score columns to dataframe"""
        print(f"Analyzing {len(df)} reviews...")
        
        sentiments = []
        for idx, row in df.iterrows():
            if row['has_text']:
                result = self.analyze(row['caption'])
                sentiments.append({
                    'sentiment': result['label'],
                    'sentiment_score': result['score']
                })
            else:
                sentiments.append({
                    'sentiment': 'NEUTRAL',
                    'sentiment_score': 0.0
                })
        
        # Add to dataframe
        df['sentiment'] = [s['sentiment'] for s in sentiments]
        df['sentiment_score'] = [s['sentiment_score'] for s in sentiments]
        
        counts = df[df['has_text']]['sentiment'].value_counts()
        print(f"Sentiment: Positive={counts.get('POSITIVE', 0)} Negative={counts.get('NEGATIVE', 0)} Neutral={counts.get('NEUTRAL', 0)}")
        
        return df

