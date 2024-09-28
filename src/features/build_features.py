import pandas as pd
import re
import os

# Function to remove HTML tags from text
def remove_html_tags(text):
    if isinstance(text, str):  # Check if text is a string
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    return text  # If not a string, return the original value

# Function to convert text to lowercase
def lower_word(text):
    return text.lower()

# Function to remove non-alphabet characters
def remove_non_words(text):
    # This regular expression keeps only letters and spaces
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

# Function to handle missing values in the 'text' column
def fill_null_text(df, fill_text="hello"):
    # Find the first index where 'text' is null
    null_index = df[df['text'].isnull()].index[0]  # Get the first null index
    
    # Replace the null value at the found index with the provided text
    df.at[null_index, 'text'] = fill_text
    
    # Verify the change by printing the updated value
    print(f"Filled index {null_index} with text: {df['text'][null_index]}")

def process_features(df):
    # Apply text transformations
    df["text"] = df["text"].astype(str)  # Ensure text is of type string
    df["text"] = df["text"].apply(remove_html_tags)  # Remove HTML tags
    df["text"] = df["text"].apply(lower_word)  # Convert text to lowercase
    df["text"] = df["text"].apply(remove_non_words)  # Remove non-alphabet characters

    # Replace sentiment categories with numerical values
    df["sentiment"] = df["sentiment"].replace({
        "neutral": 0,
        "positive": 1,
        "negative": 2,
        "suicide": 3
    })

    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Fill null text values, if any
    if df['text'].isnull().sum() > 0:
        fill_null_text(df)

    return df

def save_featured_data(df, output_path):
    # Ensure the output directory exists, then save the featured data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Featured data saved to {output_path}")

# Define file paths
input_csv_path = os.path.join("data", "processed", "processed_data.csv")  # Use the data from make_dataset.py
output_csv_path = os.path.join("data", "processed", "featured_data.csv")

# Load data from make_dataset.py output
df = pd.read_csv(input_csv_path)

# Process features
df = process_features(df)

# Save the featured data
save_featured_data(df, output_csv_path)
