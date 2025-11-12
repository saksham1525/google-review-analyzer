"""Google Review Analyzer - Streamlit App"""

import sys
import os
sys.path.append('src')

os.environ['GOOGLE_API_KEY'] = 'AIzaSyDQmi1P2Lrq7sLP7z-urjipDA2I_36GrJY'

import streamlit as st
from googlemaps import GoogleMapsScraper, clean_reviews
from sentiment import SentimentAnalyzer
from llm import GeminiAnalyzer
import pandas as pd

st.set_page_config(page_title="Review Analyzer", page_icon="📊", layout="centered")

st.markdown("""
<style>
    .main {max-width: 900px;}
    h1 {font-size: 1.8rem; font-weight: 600;}
    h3 {font-size: 1.1rem; font-weight: 500; color: #555;}
</style>
""", unsafe_allow_html=True)

st.title("Review Analyzer")
st.markdown("Analyze Google Maps reviews with AI")

# Input
url = st.text_input("Google Maps URL", placeholder="https://www.google.com/maps/place/...")
num_reviews = st.number_input("Reviews to analyze", min_value=10, max_value=500, value=100)

if st.button("Analyze", type="primary"):
    if not url:
        st.error("Please enter a URL")
    else:
        try:
            # Scraping
            with st.status("Scraping reviews...", expanded=True) as status:
                with GoogleMapsScraper(debug=False) as scraper:
                    error = scraper.sort_by(url, 0)
                    if error != 0:
                        st.error("Failed to load reviews. Check URL format.")
                        st.stop()
                    
                    all_reviews = []
                    n = 0
                    while n < num_reviews:
                        reviews = scraper.get_reviews(n)
                        if len(reviews) == 0:
                            break
                        all_reviews.extend(reviews)
                        n += len(reviews)
                    
                    df = pd.DataFrame(all_reviews)
                status.update(label=f"Scraped {len(df)} reviews", state="complete")
            
            # Processing
            with st.status("Processing sentiment..."):
                df = clean_reviews(df)
                analyzer = SentimentAnalyzer()
                df = analyzer.analyze_reviews(df)
            
            # Insights
            with st.status("Generating insights..."):
                llm = GeminiAnalyzer()
                insights = llm.generate_insights(df)
            
            st.session_state['df'] = df
            st.session_state['insights'] = insights
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")

# Display insights
if 'insights' in st.session_state:
    insights = st.session_state['insights']
    
    st.markdown("---")
    st.markdown("### Analysis Results")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Rating", f"{insights['avg_rating']:.1f}/5")
    col2.metric("Sample Size", insights['total'])
    col3.metric("With Reviews", insights['with_text'])
    
    # Sentiment
    st.markdown(f"**Sentiment:** {insights['positive']} Positive • {insights['neutral']} Neutral • {insights['negative']} Negative")
    
    # AI Insights
    st.markdown("---")
    st.markdown(insights['analysis'])
    
    # Chat
    st.markdown("---")
    st.markdown("### Ask Questions")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about the reviews"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            llm = GeminiAnalyzer()
            response = llm.ask_question(prompt, st.session_state['df'])
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
