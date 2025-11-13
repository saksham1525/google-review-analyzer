# Google Review Analyzer

AI-powered Google Maps review analyzer with RAG-based Q&A. Scrapes reviews, performs sentiment analysis, and enables semantic search across all reviews using vector embeddings.

## Overview

This tool extracts reviews from any Google Maps location, analyzes sentiment using **DistilBERT**, and generates insights using **Gemini-2.5-Flash**. Features a **RAG-powered chatbot** that searches through ALL reviews semantically to answer questions accurately.

## Tech Stack

- **Scraping**: Selenium + BeautifulSoup
- **Sentiment Analysis**: DistilBERT (Hugging Face Transformers DistilBERT)
- **LLM**: Google Gemini-2.5-Flash
- **RAG Pipeline**: LangChain + ChromaDB
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **UI**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set Google API key:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```


## Usage

### Main App (Streamlit)

Run the web interface:
```bash
streamlit run app.py
```

Enter a Google Maps URL and click "Analyze" to get:
- **Sentiment Analysis**: Average rating, sentiment distribution (Positive/Negative/Neutral)
- **AI Insights**: Key highlights, pain points, and customer tips
- **Interactive Dashboard with EDA**: Rating distribution, sentiment analysis, keyword extraction, correlations
- **RAG-Powered Q&A**: Ask questions and get accurate answers by searching ALL reviews semantically



## How It Works

1. **Scrape**: Extracts reviews from Google Maps
2. **Analyze**: DistilBERT sentiment classification on each review
3. **Embed**: Converts reviews to vector embeddings (384-dim)
4. **Store**: Saves embeddings in ChromaDB vector database
5. **RAG**: When you ask a question:
   - Query is embedded
   - Top 15 most relevant reviews retrieved via semantic search
   - LLM generates answer based on relevant context

## How to Get Google Maps URL

1. Search for a place on Google Maps
2. Click on the place to open its info panel & go to reviews tab
3. Copy the URL from the browser (contain `/maps/place/`)
4. Example: `https://www.google.com/maps/place/Restaurant+Name/@12.34,56.78...`

## Notes

- Default: 100 reviews (configurable via UI)
- Sentiment model uses 512 char limit per review
- RAG pipeline truncates reviews to 300 chars when sending to LLM
- All limits are configurable in code for productinon use

