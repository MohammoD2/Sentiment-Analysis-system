import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Data Preprocessing
def load_data(input_path):
    df = pd.read_csv(input_path)
    df_cleaned = df.dropna(subset=['text'])
    X = df_cleaned['text']
    y = df_cleaned['sentiment']
    return X, y

# Step 2: Tokenization and Padding
def tokenize_and_pad(X, max_words=5000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words, lower=True)
    tokenizer.fit_on_texts(X.values)
    X_tokenized = tokenizer.texts_to_sequences(X.values)
    X_padded = pad_sequences(X_tokenized, maxlen=max_len)
    return X_padded, tokenizer

# Step 3: LSTM Model Definition
def build_lstm_model(max_words, max_len):
    model = Sequential()
    model.add(Embedding(max_words, 100, input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(4, activation='softmax'))  # Assuming 4 sentiment classes

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Main function to load data, train the model, and save it
def train_and_save_model(train_data_path, model_save_path, tokenizer_save_path):
    # Load and preprocess data
    X, y = load_data(train_data_path)
    X_padded, tokenizer = tokenize_and_pad(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

    # Build and train model
    model = build_lstm_model(max_words=5000, max_len=100)
    model.fit(X_train, y_train, epochs=4, batch_size=50, validation_data=(X_test, y_test), verbose=2)

    # Evaluate model
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_rep)

    # Save the model and tokenizer
    model.save(model_save_path)  # Save model
    with open(tokenizer_save_path, 'wb') as file:
        pickle.dump(tokenizer, file)  # Save tokenizer

    print(f"Model and tokenizer saved successfully to {model_save_path} and {tokenizer_save_path}.")

# Define paths and run the training
train_data_path = "data/processed/featured_data.csv"  # Path from build_feature.py
model_save_path = "models/lstm_sentiment_model.h5"
tokenizer_save_path = "models/tokenizer.pkl"

train_and_save_model(train_data_path, model_save_path, tokenizer_save_path)
