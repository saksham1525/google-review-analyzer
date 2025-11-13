"""Gemini AI for review insights and Q&A"""

import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


class GeminiAnalyzer:
    """Generates insights and answers questions using Gemini"""
    
    def __init__(self, rag_pipeline=None):
        self.model = genai.GenerativeModel('gemini-2.5-flash', generation_config={'timeout': 120})
        self.rag_pipeline = rag_pipeline
        print("Gemini-2.5-Flash loaded!")
    
    def _calculate_stats(self, reviews_df):
        """Calculate review statistics"""
        text_reviews = reviews_df[reviews_df['has_text']]
        sentiment_counts = text_reviews['sentiment'].value_counts().to_dict()
        return {
            'avg_rating': reviews_df['rating'].mean(),
            'total': len(reviews_df),
            'with_text': len(text_reviews),
            'positive': sentiment_counts.get('POSITIVE', 0),
            'neutral': sentiment_counts.get('NEUTRAL', 0),
            'negative': sentiment_counts.get('NEGATIVE', 0)
        }
    
    def generate_insights(self, reviews_df):
        """Generate overall insights from reviews"""
        stats = self._calculate_stats(reviews_df)
        text_reviews = reviews_df[reviews_df['has_text']]
        
        # Build prompt
        reviews_text = "\n".join([
            f"{row['rating']}★ [{row['sentiment']}]: {row['caption'][:300]}"
            for _, row in text_reviews.head(15).iterrows()
        ])
        
        prompt = f"""Analyze these reviews briefly:

Data: {stats['total']} reviews ({stats['with_text']} with text) | Avg: {stats['avg_rating']:.1f}/5 
Sentiment: {stats['positive']} Positive, {stats['neutral']} Neutral, {stats['negative']} Negative

Sample reviews:
{reviews_text}

Provide ONLY:
1. Top positive highlights - 1-2 lines max
2. Top negative pain points - 1-2 lines max  
3. Customer tips - 1-2 lines max (advice for potential customers, what to try/order, what to avoid, best dishes mentioned)

Keep it crisp and professional. No markdown headers."""
        
        print("Generating insights...")
        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                return {**stats, 'analysis': response.text}
            except Exception as e:
                if ('503' in str(e) or 'overloaded' in str(e).lower()) and attempt < 2:
                    wait = 2 ** attempt
                    print(f"API overloaded, retry {wait}s (attempt {attempt + 1}/3)")
                    time.sleep(wait)
                else:
                    raise
    
    def ask_question(self, question, reviews_df):
        """Answer user question using RAG (searches ALL reviews)"""
        if self.rag_pipeline:
            try:
                return self.rag_pipeline.query(question, top_k=15, df_stats=self._calculate_stats(reviews_df))
            except Exception as e:
                print(f"RAG failed: {e}, using fallback")
        
        # Fallback: first 15 reviews
        text_reviews = reviews_df[reviews_df['has_text']]
        reviews_text = "\n".join([
            f"{row['rating']}⭐: {row['caption'][:300]}" 
            for _, row in text_reviews.head(15).iterrows()
        ])
        
        context = f"""You are analyzing restaurant reviews.
Total: {len(reviews_df)} | Average: {reviews_df['rating'].mean():.1f}/5

Reviews:
{reviews_text}

Question: {question}
Answer:"""
        for attempt in range(3):
            try:
                return self.model.generate_content(context).text
            except Exception as e:
                if ('503' in str(e) or 'overloaded' in str(e).lower()) and attempt < 2:
                    wait = 2 ** attempt
                    print(f"API overloaded, retry {wait}s (attempt {attempt + 1}/3)")
                    time.sleep(wait)
                else:
                    raise
