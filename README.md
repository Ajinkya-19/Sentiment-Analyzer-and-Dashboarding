# Sentiment-Analyzer-and-Dashboarding
The Sentiment Analyzer and Dashboarding is project which is combination of Sentiment analyzer and the data analysis dashboarding. the dashboard is user interactive . 
# **Sentiment Analyzer with Dashboard**

## **Project Overview**
The **Sentiment Analyzer with Dashboard** is an interactive web application that allows users to analyze customer reviews and gain actionable insights into sentiment trends. It combines **machine learning**, **data visualization**, and **natural language processing (NLP)** to process text data and present a detailed view of customer feedback.

This project is ideal for businesses looking to enhance customer satisfaction by understanding sentiment trends, identifying pain points, and analyzing feedback across product variations.

---
## **Live Demo**
The application is hosted online for easy access. **https://sentiment-analyzer-and-dashboarding.onrender.com**


## **Key Features**
### **1. Sentiment Analysis**
- Enter any customer review in the input box, and the model predicts whether the sentiment is **positive (happy)** or **negative (unhappy)**.
- Sentiment is calculated using a pre-trained machine learning model (Random Forest Classifier) and text transformation via TF-IDF vectorization.

### **2. Interactive Dashboard**
The dashboard offers:
- **Sentiment Distribution**: Visualize the proportion of happy vs. unhappy sentiments.
- **Rating Distribution**: Analyze the frequency of ratings from 1 to 5 stars.
- **Word Cloud**: Discover the most frequently used words in verified customer reviews.
- **Trends Analysis**:
  - Track changes in average ratings and sentiments over time.
  - Heatmap showing the relationship between feedback sentiment and ratings.
  - Sentiment variation by product type or other custom filters.
- **Customizable Filters**:
  - Filter reviews by ratings, product variations, and sentiment type.

### **3. Data Insights**
- Explore metrics like:
  - Total number of reviews.
  - Average customer rating.
  - Percentage of happy customers.
- Visualize review length, word counts, and weekday trends to understand customer behavior better.

---

## **Live Demo**
The application is hosted online for easy access. **https://sentiment-analyzer-and-dashboarding.onrender.com**

---

## **Technologies Used**
- **Frontend**: Streamlit (for building an interactive web UI).
- **Backend**: Python.
- **Libraries**:
  - **NLP**: NLTK (for tokenization, stemming, and stopword removal).
  - **Data Analysis**: Pandas, NumPy.
  - **Data Visualization**: Matplotlib, WordCloud,Seaborn.
  - **Machine Learning**: Scikit-learn (Random Forest Classifier, TF-IDF Vectorization).

---
## **Application Screenshots**
### **1. Sentiment Analysis Page**
#### **Positive Sentiments**
![App Overview](https://github.com/Ajinkya-19/Sentiment-Analyzer-and-Dashboarding/blob/main/Screenshot%20(124).png)
#### **Negative sentiments**
![App Overview](https://github.com/Ajinkya-19/Sentiment-Analyzer-and-Dashboarding/blob/main/Screenshot%20(125).png)

### **2. Dashboard View**
![App Overview](https://github.com/Ajinkya-19/Sentiment-Analyzer-and-Dashboarding/blob/main/Key%20matrics1.png)
![App Overview](https://github.com/Ajinkya-19/Sentiment-Analyzer-and-Dashboarding/blob/main/Key%20matrics2.png)
![App Overview](https://github.com/Ajinkya-19/Sentiment-Analyzer-and-Dashboarding/blob/main/Key%20matrics3.png)
![App Overview](https://github.com/Ajinkya-19/Sentiment-Analyzer-and-Dashboarding/blob/main/key%20matrics4.png)

---------
## **Installation and Setup**

### **1. Prerequisites**
- Python 3.8 or higher.
- Git (for cloning the repository).

### **2. Clone the Repository**
```bash
git clone https://github.com/Ajinkya-19/sentiment-analyzer-dashboard.git
cd sentiment-analyzer-dashboard
```

### **3. Install Dependencies**
Install all required Python packages:
```bash
pip install -r requirements.txt
```

### **4. Download NLTK Resources**
Ensure NLTK resources are downloaded:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### **5. Prepare Necessary Files**
Ensure the following files are present in the project directory:
- **`vectorizer.pkl`**: Pre-trained TF-IDF vectorizer for feature extraction.
- **`model.pkl`**: Logistic Regression model trained on Alexa reviews.
- **`alexa.csv`**: Dataset containing customer reviews.
- **`styles.css`**: Custom CSS for Streamlit UI.
- **`helper.py`**: Script containing helper functions.

### **6. Run the Application**
Launch the application locally:
```bash
streamlit run app.py
```

Access the application in your browser at `http://localhost:8501`.

---

## **Dataset**
The dataset used is `alexa.csv`, containing customer reviews for Amazon Alexa products. Each record includes:
- **`rating`**: Review rating (1–5 stars).
- **`feedback`**: Sentiment label (1 for happy, 0 for unhappy).
- **`variation`**: Product variation (e.g., "Black Dot").
- **`verified_reviews`**: The actual review text provided by customers.

---

## **Project Workflow**
1. **Preprocessing**:
   - Convert text to lowercase.
   - Tokenize sentences and words using NLTK.
   - Remove stopwords and non-alphanumeric characters.
   - Stem words using Porter Stemmer.

2. **Model Training**:
   - Train a Random Forest Classifier model using TF-IDF-transformed text.
   - Save the vectorizer and trained model as `.pkl` files.

3. **Application**:
   - Sentiment analysis on individual reviews.
   - Data visualizations and metrics generation using the preprocessed dataset.

---


---

## **Contributing**
We welcome contributions to enhance this project! Here's how you can help:
1. Fork this repository.
2. Clone your forked repo locally:
   ```bash
   git clone https://github.com/Ajinkya-19/sentiment-analyzer-dashboard.git
   ```
3. Create a new feature branch:
   ```bash
   git checkout -b feature-name
   ```
4. Make your changes and commit them:
   ```bash
   git commit -m "Add feature-name"
   ```
5. Push the branch to your forked repo:
   ```bash
   git push origin feature-name
   ```
6. Open a pull request to the main repository.

---

## **Future Enhancements**
- **Advanced NLP Models**: Integrate deep learning models like BERT for more accurate sentiment analysis.
- **More Visualizations**: Add graphs for deeper trend analysis.
- **Multi-language Support**: Analyze reviews written in different languages.
- **Real-time Data**: Fetch and analyze reviews directly from platforms like Amazon via API.

---

## **Author**
- **Ajinkya Chavan**
- GitHub: [Ajinkya-19](https://github.com/Ajinkya-19)
- Email: cajinkya246@gmail.com
- LinkedIn Profile: www.linkedin.com/in/ajinkya-chavan-386a79324


---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## **Acknowledgements**
- **Streamlit**: For building an interactive and user-friendly web application.
- **NLTK**: For NLP preprocessing.
- **Scikit-learn**: For machine learning capabilities.
- **Amazon Alexa Dataset**: Used for training and testing.
- **Matplotlib & WordCloud**: For data visualization.
