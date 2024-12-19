import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
from helper import load_data, preprocess_data, calculate_metrics, filter_data
from helper import sentiment_distribution, rating_distribution, generate_wordcloud, plot_trends, feedback_rating_heatmap, sentiment_by_variation, sentiment_time_series, review_length_analysis, word_count_histogram, sentiment_by_weekday

import matplotlib.pyplot as plt

# Specify the path for nltk_data
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_dir)

# Ensure NLTK resources are available
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

# Sentiment Analysis Preprocessing
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Filter out non-alphanumeric tokens and stopwords
    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load models for sentiment analysis
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Load and preprocess data for analysis dashboard
data = load_data("alexa.csv")
data = preprocess_data(data)

# Apply custom CSS for styling
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Mode", ["Home", "Sentiment Analyzer", "Dashboard"])

# Main UI
if app_mode == "Dashboard":
    # Hero Section
    st.markdown("""
    <div class="hero">
        <h1>Alexa Sentiment Dashboard</h1>
        <p>Explore trends and insights from Alexa reviews.</p>
    </div>
    """, unsafe_allow_html=True)

    # Filters and metrics
    st.sidebar.title("Filters")
    rating_filter = st.sidebar.multiselect("Select Ratings", [1, 2, 3, 4, 5])
    feedback_filter = st.sidebar.multiselect("Select Feedback Sentiment", ["Happy", "Unhappy"])
    variation_filter = st.sidebar.multiselect("Select Product Variation", data['variation'].unique())

    filtered_data = filter_data(data,
                                rating=rating_filter,
                                feedback=[1 if f == "Happy" else 0 for f in feedback_filter] if feedback_filter else None,
                                variation=variation_filter)

    total_reviews, avg_rating, happy_percentage = calculate_metrics(filtered_data)
    st.markdown("<h2>Key Metrics</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metrics-box'><h3>Total Reviews</h3><p>{total_reviews}</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metrics-box'><h3>Average Rating</h3><p>{avg_rating:.1f}</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metrics-box'><h3>Happy Customers (%)</h3><p>{happy_percentage:.1f}%</p></div>", unsafe_allow_html=True)

    # Visualizations
    st.markdown("### Sentiment Distribution")
    sentiment_counts = sentiment_distribution(filtered_data)
    st.bar_chart(sentiment_counts)

    st.markdown("### Rating Distribution")
    rating_counts = rating_distribution(filtered_data)
    st.bar_chart(rating_counts)

    # Word cloud for verified reviews
    st.markdown("### Verified Reviews Word Cloud")
    wordcloud = generate_wordcloud(" ".join(filtered_data['verified_reviews'].dropna()))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Trends
    st.markdown("### Rating and Sentiment Trends Over Time")
    trend_data = plot_trends(filtered_data)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trend_data['date'], trend_data['rating'], label='Average Rating', color='blue')
    ax.plot(trend_data['date'], trend_data['feedback'], label='Average Feedback', color='green')
    ax.set_title('Trends Over Time')
    ax.legend()
    st.pyplot(fig)

    st.markdown("<h2>Feedback vs. Rating Heatmap</h2>", unsafe_allow_html=True)
    heatmap_fig = feedback_rating_heatmap(filtered_data)
    st.pyplot(heatmap_fig)

    st.markdown("<h2>Sentiment by Product Variation</h2>", unsafe_allow_html=True)
    variation_fig = sentiment_by_variation(filtered_data)
    st.pyplot(variation_fig)

    st.markdown("<h2>Sentiment Trends Over Time</h2>", unsafe_allow_html=True)
    time_series_fig = sentiment_time_series(filtered_data)
    st.pyplot(time_series_fig)

elif app_mode == "Sentiment Analyzer":
    # Hero Section
    st.markdown("""
    <div class="hero">
        <h1>Sentiment Analyzer</h1>
        <p>Enter a review to predict its sentiment.</p>
    </div>
    """, unsafe_allow_html=True)

    # Input area for sentiment analysis
    input_review = st.text_area("Enter the Review", placeholder="Type your review here...")

    if st.button("Analyze Sentiment"):
        if input_review.strip():
            # Preprocess and predict
            transformed_review = transform_text(input_review)
            vector_input = tfidf.transform([transformed_review])
            result = model.predict(vector_input)[0]

            # Display result
            if result == 1:
                st.markdown("""
                <div class="metrics-box" style="background: #2ecc71;">
                    <h3>Happy Sentiment</h3>
                    <p>😊 The review indicates positive sentiment.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metrics-box" style="background: #e74c3c;">
                    <h3>Unhappy Sentiment</h3>
                    <p>😞 The review indicates negative sentiment.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a review before analyzing.")

elif app_mode == "Home":
    # Home Tab: Sentiment Analyzer Overview
    st.title("Welcome to the Sentiment Analyzer")

    st.markdown("""
## About the Project

The **Sentiment Analyzer with Dashboard** is a powerful tool designed to analyze customer reviews and provide insightful visualizations based on their sentiments. 
This project enables users to understand customer feedback more effectively by predicting the sentiment (positive or negative) of reviews and offering a comprehensive dashboard for further analysis.

### Features:
- **Sentiment Analysis:** Predict whether a given review expresses a positive or negative sentiment.
- **Interactive Dashboard:** Explore trends, visualize data, and analyze key metrics like sentiment distribution, rating distribution, and customer happiness percentage.
- **Word Cloud & Trends:** Gain insights into common keywords and observe sentiment trends over time.

--- 

## Data Source
The dashboard is built using real customer reviews for Amazon Alexa products. These reviews have been processed and visualized to uncover meaningful patterns and insights.

### Author:
Ajinkya Chavan

### Github:
https://github.com/Ajinkya-19
""")
