import pandas as pd
import os

def process_and_save_data(train_path, suicide_detection_path, output_path):
    # Load the first dataset
    data = pd.read_csv(train_path, encoding='ISO-8859-1')
    data = data[["text", "sentiment"]]

    # Load the second dataset and filter the 'suicide' class
    new_df = pd.read_csv(suicide_detection_path, encoding='ISO-8859-1')
    new_df = new_df[new_df['class'] == 'suicide']
    new_df.reset_index(drop=True, inplace=True)

    # Rename the column and clean the dataframe
    new_df["sentiment"] = new_df["class"]
    new_df = new_df.drop(columns=["class", "Unnamed: 0"])
    new_df.reset_index(drop=True, inplace=True)

    # Concatenate the two dataframes
    df = pd.concat([
        data[['text', 'sentiment']],
        new_df[['text', 'sentiment']].sample(10000)
    ], ignore_index=True)

    # Ensure the output directory exists, then save the processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Processed data saved to {output_path}")
    return df

# Define file paths using os.path.join for cross-platform compatibility
train_csv_path = os.path.join("data", "raw", "train.csv")
suicide_detection_csv_path = os.path.join("data", "raw", "Suicide_Detection.csv")
output_csv_path = os.path.join("data", "processed", "processed_data.csv")

# Run the function
process_and_save_data(train_csv_path, suicide_detection_csv_path, output_csv_path)
