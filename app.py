"""Google Review Analyzer - Streamlit App"""

import sys, os
sys.path.append('src')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
if 'GOOGLE_API_KEY' not in os.environ:
    raise ValueError("Google API key not found")

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

# Display functions
def show_metrics(df):
    st.markdown("### Analysis Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Rating", f"{df['rating'].mean():.1f}/5")
    col2.metric("Sample Size", len(df))
    col3.metric("With Reviews", len(df[df['has_text']]))

def show_sentiment(sentiment_counts):
    st.markdown("**Sentiment Distribution**")
    col1, col2 = st.columns(2)
    col1.metric("😊 Positive", sentiment_counts.get('POSITIVE', 0))
    col2.metric("😞 Negative", sentiment_counts.get('NEGATIVE', 0))

def show_dashboard(df):
    st.markdown("---")
    st.markdown("### Data Insights Dashboard")
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "😊 Sentiment", "📝 Text Analysis"])
    
    with tab1:
        st.plotly_chart(plot_rating_distribution(df), width='stretch')
        if fig := plot_sentiment_proportion_by_rating(df):
            st.plotly_chart(fig, width='stretch')
    
    with tab2:
        st.pyplot(plot_sentiment_pie(df))
        col1, col2 = st.columns(2)
        with col1:
            if kw_pos := plot_top_keywords(df, 'POSITIVE'):
                st.pyplot(kw_pos)
        with col2:
            if kw_neg := plot_top_keywords(df, 'NEGATIVE'):
                st.pyplot(kw_neg)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            if fig := plot_text_length_distribution(df):
                st.plotly_chart(fig, width='stretch')
        with col2:
            st.pyplot(plot_correlation_heatmap(df))

def show_insights(insights):
    st.markdown("---")
    st.markdown("### 🤖 AI Insights")
    for line in insights['analysis'].split('\n\n'):
        if line.strip():
            st.markdown(line)

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
                    print("Scraping reviews...")
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
                        for review in reviews:
                            st.write(f"**⭐{int(review.get('rating', 0))}/5**: {review.get('caption', 'No text')}")
                        all_reviews.extend(reviews)
                        n += len(reviews)
                    
                    df = pd.DataFrame(all_reviews)
                status.update(label=f"Scraped {len(df)} reviews", state="complete")
            
            df = clean_reviews(df)
            show_metrics(df)

            # Processing sentiment
            with st.spinner("Processing sentiment..."):
                analyzer = SentimentAnalyzer()
                df = analyzer.analyze_reviews(df)

            sentiment_counts = df[df['has_text']]['sentiment'].value_counts().to_dict()
            show_sentiment(sentiment_counts)
            show_dashboard(df)
            
            # Build Vector Store for RAG
            st.markdown("---")
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
            
            # Generate and display insights
            with st.spinner("Generating AI insights..."):
                llm = GeminiAnalyzer(rag_pipeline=rag_pipeline)
                insights = llm.generate_insights(df)
            
            show_insights(insights)
            print("Processing complete!")
            
            # Store for chat and reruns (clear old analysis, keep new)
            st.session_state.clear()
            st.session_state['df'] = df
            st.session_state['llm'] = llm
            st.session_state['sentiment_counts'] = sentiment_counts
            st.session_state['insights'] = insights
            st.session_state['messages'] = []  # Initialize empty chat for new analysis
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

# Redisplay on chat rerun
elif 'df' in st.session_state:
    show_metrics(st.session_state['df'])
    show_sentiment(st.session_state['sentiment_counts'])
    show_dashboard(st.session_state['df'])
    show_insights(st.session_state['insights'])

# Chat interface (always at end)
if 'df' in st.session_state:
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
            llm = st.session_state.get('llm', GeminiAnalyzer())
            print("Answering question using RAG: All reviews searched" if 'llm' in st.session_state else "Fallback mode")
            response = llm.ask_question(prompt, st.session_state['df'])
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
