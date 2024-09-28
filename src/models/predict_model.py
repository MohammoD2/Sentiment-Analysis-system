import re 
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Step 1: Load the saved LSTM model and tokenizer
def load_model_and_tokenizer(model_path, tokenizer_path):
    model = load_model(model_path)

    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)

    return model, tokenizer

# Step 2: Preprocess the input comment (you can adjust this based on your actual preprocessing)
def preprocessing(comment):
    comment = comment.lower()  # Convert to lowercase
    comment = re.sub(r'[^a-zA-Z\s]', '', comment)  # Remove non-alphabetic characters
    return comment

# Step 3: Function for predicting sentiment using the LSTM model
def predict_sentiment(comment, model, tokenizer, max_len=100):
    # Preprocess the input comment
    preprocessed_comment = preprocessing(comment)
    
    # Tokenize and pad the comment
    comment_sequence = tokenizer.texts_to_sequences([preprocessed_comment])
    comment_padded = pad_sequences(comment_sequence, maxlen=max_len)  # Use the same max_len as during training
    
    # Make a prediction using the LSTM model
    prediction = np.argmax(model.predict(comment_padded), axis=1)[0]
    
    return prediction

# Step 4: Main function for prediction
def main():
    model_path = "models/lstm_sentiment_model.h5"
    tokenizer_path = "models/tokenizer.pkl"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

    # Input comment for prediction
    input_comment = input("Enter a comment for sentiment prediction: ")
    
    # Get prediction
    pred = predict_sentiment(input_comment, model, tokenizer)
    
    # Map prediction to sentiment labels
    if pred == 0:
        print("neutral comment")
    elif pred == 1:
        print("positive comment")
    elif pred == 2:
        print("negative comment")
    else:
        print("suicide comment")

if __name__ == "__main__":
    main()
