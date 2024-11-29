# Sentify Chatbot

This project is a sentiment analysis application built using TextBlob and Streamlit. The application allows users to input text, analyzes the sentiment, and provides feedback as Positive, Negative, or Neutral. Additionally, the model evaluates a dataset of Amazon reviews to showcase performance metrics.


## Features

- Interactive Sentiment Analysis: Users can input custom text to analyze its sentiment.

- Text Preprocessing: Includes tokenization, stopword removal, and lemmatization for clean text processing.

- Sentiment Classification: Identifies sentiment as:
    - Positive 😊
    - Negative 😞
    - Neutral 😐

- Performance Metrics: Displays accuracy and a classification report on the Amazon review dataset.


## Tech Stack

- Python: Core language for the application.

- TextBlob: Library for sentiment analysis.

- NLTK: Used for text preprocessing.

- Streamlit: Framework for building the user interface.


## Installation

<!--start code-->
#### 1. Install required libraries:
    pip install pandas textblob streamlit nltk
<!--end code-->

#### 2. Download NLTK datasets (required for preprocessing):

import nltk

nltk.download('stopwords')

nltk.download('punkt')

nltk.download('wordnet')


## How it Works

### 1. Data Loading

- The app uses an Amazon review dataset as an example for testing and evaluation. This dataset is loaded directly from an online source using Pandas.
- The dataset contains text reviews which are analyzed for sentiment.

### 2. Text Preprocessing

The reviews are processed to clean and normalize the text for better analysis:

- Tokenization: Breaks the text into individual words (tokens).
- Stopword Removal: Removes common words like "the," "is," and "and" that don't contribute to sentiment.
- Lemmatization: Converts words to their base forms (e.g., "running" → "run").

### 3. Sentiment Analysis

- TextBlob is used to calculate the polarity of the text:
    -   Polarity ranges from -1 (completely negative) to 1 (completely positive).

- The polarity score is mapped to three sentiment categories:
    - Positive 😊: If the polarity is greater than 0.1.
    - Negative 😞: If the polarity is less than -0.1.
    - Neutral 😐: If the polarity equals 0.

### 4. Model Evaluation

- The dataset is split into training and testing sets to evaluate the app's sentiment detection.
- TextBlob directly predicts sentiment for the test set, and performance is measured using:
    - Accuracy: How many predictions match the actual sentiment.
    - Classification Report: Includes precision, recall, and F1-score for sentiment categories.

### 5. User Interface

<!--start code-->
#### Built with Streamlit, the app provides an interactive interface:
    streamlit run app.py
<!--end code-->

- User Input: Users can type any text into the input box.
- Real-Time Sentiment Detection: The input is processed and analyzed instantly using TextBlob.
- Output: Displays the sentiment as Positive, Negative, or Neutral, along with a corresponding emoji.


## Example Outputs

### User Input:
- "I love this product!"
- Sentiment Output: "Positive 😊"

### User Input:
- "This is not what I expected."
- Sentiment Output: "Negative 😞"

### User Input:
- "It's okay, nothing special."
- Sentiment Output: "Neutral 😐"

