from textblob import TextBlob
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import nltk

# download only for first time

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('vader_lexicon')

@st.cache_resource
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')
    return df

df = load_data()

# Preprocessing function
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Return processed text
    return ' '.join(lemmatized_tokens)

# Preprocess the dataset
df['reviewText'] = df['reviewText'].astype(str)  # Ensure text column is string
df['processedText'] = df['reviewText'].apply(preprocess_text)


# Assign sentiments for training
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 1
    elif polarity < -0.1:
        return -1
    else:
        return 0


df['sentiment'] = df['reviewText'].apply(get_sentiment)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['processedText'], df['sentiment'], test_size=0.2, random_state=42)

# Train/Test predictions (TextBlob doesn't train; it directly analyzes)
y_pred = X_test.apply(lambda x: get_sentiment(x))
accuracy = accuracy_score(y_test, y_pred)

