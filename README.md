# Google Review Analyzer

Scrapes Google Maps reviews, provides AI-powered sentiment analysis & insights, including Q&A chatbot functionality using Gemini.

## Overview

This tool extracts reviews from any Google Maps location, analyzes sentiment using DistilBERT, and generates insights using Gemini-2.5-Flash. Includes an interactive chatbot to ask questions about the reviews.

## Tech Stack

- **Scraping**: Selenium with Chrome WebDriver
- **Sentiment Analysis**: Hugging Face Transformers (DistilBERT)
- **LLM**: Google Gemini-2.5-Flash (free tier)
- **UI**: Streamlit
- **Data Processing**: Pandas

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
- Average rating and sample size
- Sentiment distribution
- AI-generated insights (highlights, pain points, customer tips)
- Interactive chatbot for Q&A



## How to Get Google Maps URL

1. Search for a place on Google Maps
2. Click on the place to open its info panel & go to reviews tab
3. Copy the URL from the browser (should contain `/maps/place/`)
4. Example: `https://www.google.com/maps/place/Restaurant+Name/@12.34,56.78...`

## Notes

- Scraper uses the first 100 reviews by default (configurable in app)
- Sentiment model truncates reviews to 512 characters for classification
- LLM uses first 15 reviews (300 chars each) for insights and chat
- All limits are configurable in code for production use

