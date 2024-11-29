import streamlit as st
from main import *
# from vader import *

# Streamlit Interface
# st.title("Sentiment Analyzer Chatbot")
# st.write("This app analyzes the sentiment of user messages using a trained model based on Amazon reviews dataset.")

# # Input form for user message
# user_input = st.text_input("Enter your message:")
# if st.button("Analyze Sentiment"):
#     if user_input:
#         # Preprocess user input
#         processed_input = preprocess_text(user_input)
#         # Vectorize input
#         input_vec = vectorizer.transform([processed_input])
#         # Predict sentiment
#         prediction = model.predict(input_vec)[0]
#         # Display result
#         sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
#         st.write(f"Sentiment: {sentiment}")
#     else:
#         st.write("Please enter a message to analyze.")

# # Display model evaluation metrics
# st.subheader("Model Performance")
# st.write(f"Accuracy: {accuracy:.2f}")
# st.text("Classification Report:")
# st.text(classification_report(y_test, y_pred))

st.title("Sentiment Analyzer Chatbot")
st.write("This app analyzes the sentiment of user messages using TextBlob sentiment analysis.")

# Input form for user message
user_input = st.text_input("Enter your message:")
if st.button("Analyze Sentiment"):
    if user_input:
        # Analyze sentiment
        sentiment = get_sentiment(user_input)
        sentiment_text = "Positive 😊" if sentiment == 1 else "Negative 😞" if sentiment == -1 else "Neutral 😐"
        st.write(f"Sentiment: {sentiment_text}")
    else:
        st.write("Please enter a message to analyze.")

# Display model evaluation metrics
st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))