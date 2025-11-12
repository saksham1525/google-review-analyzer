"""Google Review Analyzer - Streamlit App"""

import sys
import os
sys.path.append('src')

os.environ['GOOGLE_API_KEY'] = 'AIzaSyDQmi1P2Lrq7sLP7z-urjipDA2I_36GrJY'

import streamlit as st
from googlemaps import GoogleMapsScraper, clean_reviews
from sentiment import SentimentAnalyzer
from llm import GeminiAnalyzer
from visualizations import *
from embeddings import EmbeddingGenerator
from vector_store import ReviewVectorStore
from rag_pipeline import RAGPipeline
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
            
            # Build Vector Store for RAG
            with st.status("Building knowledge base...", expanded=True) as status:
                st.write("Generating embeddings...")
                embedder = EmbeddingGenerator()
                embeddings, text_reviews = embedder.embed_reviews(df)
                
                st.write("Storing in vector database...")
                vector_store = ReviewVectorStore(persist_directory="./chroma_db")
                vector_store.create_collection("reviews")
                vector_store.add_reviews(embeddings, text_reviews)
                
                st.write("Initializing RAG pipeline...")
                rag_pipeline = RAGPipeline(vector_store, embedder)
                
                stats = vector_store.get_collection_stats()
                status.update(
                    label=f"Knowledge base ready ({stats['count']} reviews indexed)", 
                    state="complete"
                )
            
            # Insights
            with st.status("Generating insights..."):
                llm = GeminiAnalyzer(rag_pipeline=rag_pipeline)
                insights = llm.generate_insights(df)
            
            # Store in session
            st.session_state['df'] = df
            st.session_state['insights'] = insights
            st.session_state['rag_pipeline'] = rag_pipeline
            st.session_state['llm'] = llm
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
    
    # Dashboard
    st.markdown("---")
    st.markdown("### Data Insights Dashboard")
    
    df_viz = st.session_state['df']
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "😊 Sentiment", "📝 Text Analysis"])
    
    with tab1:
        st.plotly_chart(plot_rating_distribution(df_viz), use_container_width=True)
        if fig := plot_sentiment_proportion_by_rating(df_viz):
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.pyplot(plot_sentiment_pie(df_viz))
        col1, col2 = st.columns(2)
        with col1:
            if kw_pos := plot_top_keywords(df_viz, 'POSITIVE'):
                st.pyplot(kw_pos)
        with col2:
            if kw_neg := plot_top_keywords(df_viz, 'NEGATIVE'):
                st.pyplot(kw_neg)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            if fig := plot_text_length_distribution(df_viz):
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.pyplot(plot_correlation_heatmap(df_viz))
    
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
            # Use RAG-enhanced LLM if available
            if 'llm' in st.session_state:
                llm = st.session_state['llm']
            else:
                llm = GeminiAnalyzer()
            
            response = llm.ask_question(prompt, st.session_state['df'])
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
