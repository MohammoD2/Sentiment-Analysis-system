import streamlit as st
import pickle
import re

# Preprocessing function
def preprocessing(text):
    # Check if text is a string
    if isinstance(text, str):
        # Remove HTML tags
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text)  # Correctly assign the cleaned text
        
        # Convert text to lowercase
        text = text.lower()
        
        # Remove non-alphabetic characters (keep spaces)
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return cleaned_text  # Return the final cleaned text
    else:
        return text  # Return the original text if it's not a string

# Load the model and vectorizer
with open('models/clf_loggi_reg.pkl', 'rb') as model_file:
    clf_loggi_reg = pickle.load(model_file)

with open('models/tfidf.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Prediction function
def predict_sentiment(comment):
    preprocessed_comment = preprocessing(comment)
    comment_list = [preprocessed_comment]  # Wrap the preprocessed comment in a list
    comment_vector = tfidf.transform(comment_list)
    prediction = clf_loggi_reg.predict(comment_vector)[0]
    return prediction

st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")

# Streamlit app layout
st.title("Sentiment Analysis")

# Input text from user
input_comment = st.text_area("Enter a comment for sentiment analysis:")

# Button to run prediction
if st.button('Analyze Sentiment'):
    # Check if the input comment is empty
    if not input_comment.strip():
        st.error("Please enter a comment to analyze.", icon="❌")
    else:
        # If input is not empty, proceed with prediction
        pred = predict_sentiment(input_comment)
        if pred == 0:
            st.success("**Neutral Comment**", icon="😊")
        elif pred == 1:
            st.success("**Positive Comment**", icon="😃")
        elif pred == 2:
            st.warning("**Negative Comment**", icon="😖")
        else:
            st.warning("**Suicide Comment**", icon="😢")
