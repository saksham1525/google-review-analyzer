"""Gemini AI for review insights and Q&A"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


class GeminiAnalyzer:
    """Generates insights and answers questions using Gemini"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        print("Gemini-2.5-Flash loaded!")
    
    def generate_insights(self, reviews_df):
        """Returns dict with metrics and AI-generated analysis"""
        text_reviews = reviews_df[reviews_df['has_text']]
        
        # Calculate metrics
        avg_rating = reviews_df['rating'].mean()
        total = len(reviews_df)
        with_text = len(text_reviews)
        sentiment_counts = text_reviews['sentiment'].value_counts().to_dict()
        
        # Create prompt
        prompt = f"""Analyze these reviews briefly:

Data:
- Total: {total} reviews
- With text: {with_text} reviews
- Average rating: {avg_rating:.1f}/5
- Sentiment: {sentiment_counts}

Sample reviews:
"""
        # first 15 reviews with 300 char limit per review due to rate & token limits (increase in production)
        for idx, row in text_reviews.head(15).iterrows():
            prompt += f"\n{row['rating']}★ [{row['sentiment']}]: {row['caption'][:300]}"
        
        prompt += """

Provide ONLY:
1. Top positive highlights - 1-2 lines max, common positive themes/keywords
2. Top negative pain points - 1-2 lines max, frequent complaints/keywords  
3. Customer tips - 1-2 lines max, what to try/order, what to avoid, best dishes mentioned

Keep it crisp and professional. No markdown headers. Frame #3 as advice for potential customers visiting this place."""
        
        print("Generating insights...")
        response = self.model.generate_content(prompt)
        
        # Format output
        insights = {
            'avg_rating': avg_rating,
            'total': total,
            'with_text': with_text,
            'positive': sentiment_counts.get('POSITIVE', 0),
            'neutral': sentiment_counts.get('NEUTRAL', 0),
            'negative': sentiment_counts.get('NEGATIVE', 0),
            'analysis': response.text
        }
        return insights
    
    def ask_question(self, question, reviews_df):
        """Answers user question by sending it to Gemini with top 15 reviews as context"""
        text_reviews = reviews_df[reviews_df['has_text']]
        
        # Build context
        context = f"""You are analyzing restaurant reviews. Here's the data:

Total: {len(reviews_df)} reviews
Average Rating: {reviews_df['rating'].mean():.1f}/5

Reviews:
"""
        # first 15 reviews with 300 char limit per review due to rate & token limits (increase in production)
        for idx, row in text_reviews.head(15).iterrows():
            context += f"\n{row['rating']}⭐: {row['caption'][:300]}"
        
        context += f"\n\nQuestion: {question}\n\nAnswer based on the reviews above:"
        
        response = self.model.generate_content(context)
        return response.text

