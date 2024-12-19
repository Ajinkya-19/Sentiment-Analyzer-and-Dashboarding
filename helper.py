import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Load dataset
def load_data(filepath):
    return pd.read_csv(filepath)

# Preprocess dataset
def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])  # Convert date to datetime
    df['sentiment'] = df['feedback'].map({0: 'Unhappy', 1: 'Happy'})  # Map feedback
    return df

# Calculate overall metrics
def calculate_metrics(df):
    total_reviews = len(df)
    avg_rating = df['rating'].mean()
    happy_percentage = (df[df['feedback'] == 1].shape[0] / total_reviews) * 100
    return total_reviews, avg_rating, happy_percentage

# Filter data
def filter_data(df, start_date=None, end_date=None, rating=None, feedback=None, variation=None):
    filtered_df = df.copy()
    if start_date:
        filtered_df = filtered_df[filtered_df['date'] >= start_date]
    if end_date:
        filtered_df = filtered_df[filtered_df['date'] <= end_date]
    if rating:
        filtered_df = filtered_df[filtered_df['rating'].isin(rating)]
    if feedback:
        filtered_df = filtered_df[filtered_df['feedback'].isin(feedback)]
    if variation:
        filtered_df = filtered_df[filtered_df['variation'].isin(variation)]
    return filtered_df
# Generate sentiment distribution
def sentiment_distribution(df):
    return df['sentiment'].value_counts()

# Generate rating distribution
def rating_distribution(df):
    return df['rating'].value_counts()

# Generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(background_color='white', max_words=100, contour_width=3, contour_color='steelblue')
    return wordcloud.generate(text)

# Plot trends
def plot_trends(df):
    trend_data = df.groupby(df['date'].dt.to_period('M')).agg({'rating': 'mean', 'feedback': 'mean'}).reset_index()
    trend_data['date'] = trend_data['date'].dt.to_timestamp()
    return trend_data


import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap for feedback vs. rating
def feedback_rating_heatmap(df):
    heatmap_data = df.groupby(['rating', 'sentiment']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="coolwarm", ax=ax)
    ax.set_title("Feedback vs. Rating Heatmap")
    return fig

# Bar chart for sentiment by variation
def sentiment_by_variation(df):
    variation_data = df.groupby(['variation', 'sentiment']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    variation_data.plot(kind='bar', stacked=True, ax=ax, colormap="Set2")
    ax.set_title("Sentiment by Product Variation")
    ax.set_xlabel("Product Variation")
    ax.set_ylabel("Count")
    return fig

# Time series analysis of sentiment
def sentiment_time_series(df):
    time_data = df.groupby([df['date'].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0)
    time_data.index = time_data.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 6))
    time_data.plot(ax=ax)
    ax.set_title("Sentiment Trends Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    return fig

# Boxplot for review length by sentiment
def review_length_analysis(df):
    df['review_length'] = df['verified_reviews'].str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x='sentiment', y='review_length', ax=ax, palette="Set2")
    ax.set_title("Review Length by Sentiment")
    return fig

# Word count distribution
def word_count_histogram(df):
    df['word_count'] = df['verified_reviews'].str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['word_count'], kde=True, bins=20, ax=ax, color="skyblue")
    ax.set_title("Word Count Distribution in Verified Reviews")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    return fig

# Average sentiment by weekday
def sentiment_by_weekday(df):
    df['weekday'] = df['date'].dt.day_name()
    weekday_sentiment = df.groupby('weekday')['feedback'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    weekday_sentiment.plot(kind='bar', ax=ax, color="steelblue")
    ax.set_title("Average Sentiment by Weekday")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Average Sentiment (0=Unhappy, 1=Happy)")
    return fig
