"""Visualization functions for review analysis dashboard"""

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import re

SENTIMENT_COLORS = {'POSITIVE': '#00CC96', 'NEGATIVE': '#EF553B', 'NEUTRAL': '#636EFA'}

def plot_rating_distribution(df):
    """Histogram showing distribution of star ratings"""
    rating_counts = df['rating'].value_counts().sort_index()
    
    fig = px.bar(rating_counts, title='Rating Distribution',
                 labels={'value': 'Number of Reviews', 'index': 'Star Rating'})
    fig.update_traces(marker_color='#636EFA')
    fig.update_xaxes(tick0=1, dtick=1, title='Star Rating')
    fig.update_yaxes(title='Number of Reviews')
    fig.update_layout(bargap=0.1, showlegend=False)
    return fig


def plot_sentiment_pie(df):
    """Pie chart of sentiment proportions"""
    sentiment_counts = df[df['has_text']]['sentiment'].value_counts()
    colors = [SENTIMENT_COLORS.get(s, '#636EFA') for s in sentiment_counts.index]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90, textprops={'fontsize': 12})
    ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    return fig


def plot_sentiment_proportion_by_rating(df):
    """Bar chart showing percentage of positive/negative reviews per rating"""
    df_filtered = df[(df['has_text']) & (df['rating'].notna()) & (df['sentiment'].notna())]
    
    if len(df_filtered) == 0:
        return None
    
    # Calculate percentages
    grouped = df_filtered.groupby(['rating', 'sentiment']).size().unstack(fill_value=0)
    for col in ['POSITIVE', 'NEGATIVE']:
        if col not in grouped.columns:
            grouped[col] = 0
    
    percentages = (grouped.div(grouped.sum(axis=1), axis=0) * 100)[['POSITIVE', 'NEGATIVE']]
    plot_df = percentages.reset_index().melt(id_vars='rating', var_name='sentiment', value_name='percentage')
    
    fig = px.bar(plot_df, x='rating', y='percentage', color='sentiment',
                 title='Sentiment Proportion by Rating', barmode='group',
                 color_discrete_map={'POSITIVE': '#00CC96', 'NEGATIVE': '#EF553B'})
    fig.update_xaxes(tick0=1, dtick=1, title='Star Rating')
    fig.update_yaxes(title='Percentage (%)', range=[0, 100])
    return fig


def plot_text_length_distribution(df):
    """Histogram of review text lengths"""
    text_df = df[(df['has_text']) & (df['text_length'] > 0)]
    
    if len(text_df) == 0:
        return None
    
    fig = px.histogram(text_df, x='text_length', title='Review Length Distribution', nbins=30)
    fig.update_traces(marker_color='#636EFA')
    fig.update_xaxes(title='Number of Characters')
    fig.update_yaxes(title='Number of Reviews')
    fig.update_layout(showlegend=False, bargap=0.05)
    return fig


def plot_correlation_heatmap(df):
    """Correlation heatmap for numeric features"""
    df_text = df[df['has_text']].copy()
    df_text['sentiment_binary'] = (df_text['sentiment'] == 'POSITIVE').astype(int)
    
    corr_data = df_text[['rating', 'sentiment_binary', 'sentiment_score', 'text_length', 'n_review_user']].corr()
    
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, ax=ax,
                fmt='.2f', cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix', fontsize=14, pad=20)
    return fig


def plot_top_keywords(df, sentiment_filter, top_n=10):
    """Bar chart of top keywords for specific sentiment"""
    text_data = df[df['sentiment'] == sentiment_filter]['caption']
    if len(text_data) == 0:
        return None
    
    # Extract keywords (words longer than 3 chars, excluding common words)
    stop_words = {'the', 'and', 'was', 'for', 'with', 'this', 'that', 'but', 'from', 'very', 'have', 'had', 'has', 'are', 'were'}
    all_text = ' '.join(text_data.dropna()).lower()
    words = re.findall(r'\b[a-z]{4,}\b', all_text)
    words = [w for w in words if w not in stop_words]
    
    word_counts = Counter(words).most_common(top_n)
    if not word_counts:
        return None
    
    keywords, counts = zip(*word_counts)
    color = SENTIMENT_COLORS.get(sentiment_filter, '#636EFA')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(keywords, counts, color=color, alpha=0.8)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Keyword', fontsize=12)
    ax.set_title(f'Top Keywords in {sentiment_filter.capitalize()} Reviews', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    return fig

