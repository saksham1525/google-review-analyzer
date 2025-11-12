"""Gemini AI for review insights and Q&A"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


class GeminiAnalyzer:
    """Generates insights and answers questions using Gemini"""
    
    def __init__(self, rag_pipeline=None):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.rag_pipeline = rag_pipeline  # Optional RAG pipeline for enhanced Q&A
        print("Gemini-2.5-Flash loaded!")
    
    def generate_insights(self, reviews_df):
        """Generate overall insights from reviews"""
        text_reviews = reviews_df[reviews_df['has_text']]
        
        # Calculate metrics
        avg_rating = reviews_df['rating'].mean()
        total = len(reviews_df)
        with_text = len(text_reviews)
        sentiment_counts = text_reviews['sentiment'].value_counts().to_dict()
        
        # Create prompt with sample reviews
        prompt = f"""Analyze these reviews briefly:

Data:
- Total: {total} reviews
- With text: {with_text} reviews
- Average rating: {avg_rating:.1f}/5
- Sentiment: {sentiment_counts}

Sample reviews:
"""
        
        # Use first 15 reviews for quick insights
        for idx, row in text_reviews.head(15).iterrows():
            prompt += f"\n{row['rating']}★ [{row['sentiment']}]: {row['caption'][:300]}"
        
        prompt += """

Provide ONLY:
1. Top positive highlights - 1-2 lines max, common positive themes/keywords
2. Top negative pain points - 1-2 lines max, frequent complaints/keywords  
3. Customer tips - 1-2 lines max, what to try/order, what to avoid, best dishes mentioned

Keep it crisp and professional. No markdown headers. Frame #3 as advice for potential customers."""
        
        print("Generating insights...")
        response = self.model.generate_content(prompt)
        
        return {
            'avg_rating': avg_rating,
            'total': total,
            'with_text': with_text,
            'positive': sentiment_counts.get('POSITIVE', 0),
            'neutral': sentiment_counts.get('NEUTRAL', 0),
            'negative': sentiment_counts.get('NEGATIVE', 0),
            'analysis': response.text
        }
    
    def ask_question(self, question, reviews_df):
        """Answer user question using RAG (searches ALL reviews semantically)"""
        if self.rag_pipeline:
            try:
                stats = {
                    'total': len(reviews_df),
                    'avg_rating': reviews_df['rating'].mean(),
                    'positive': len(reviews_df[reviews_df['sentiment'] == 'POSITIVE']),
                    'neutral': len(reviews_df[reviews_df['sentiment'] == 'NEUTRAL']),
                    'negative': len(reviews_df[reviews_df['sentiment'] == 'NEGATIVE'])
                }
                return self.rag_pipeline.query(question, top_k=15, df_stats=stats)
            except Exception as e:
                print(f"RAG failed: {e}, using fallback")
        
        # Fallback: first 15 reviews
        text_reviews = reviews_df[reviews_df['has_text']]
        context = f"""You are analyzing restaurant reviews.

Total: {len(reviews_df)} reviews | Average: {reviews_df['rating'].mean():.1f}/5

Reviews:
"""
        for idx, row in text_reviews.head(15).iterrows():
            context += f"\n{row['rating']}⭐: {row['caption'][:300]}"
        
        context += f"\n\nQuestion: {question}\nAnswer:"
        return self.model.generate_content(context).text
