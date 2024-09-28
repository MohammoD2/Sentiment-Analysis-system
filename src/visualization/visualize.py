# visualization.py
import pandas as pd
import plotly.express as px

def visualize_sentiment_distribution(data_path='data/processed/featured_data.csv'):
    # Load processed data
    df = pd.read_csv(data_path)
    
    # Sentiment Distribution Pie Chart
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']

    # Create Pie Chart
    fig = px.pie(sentiment_counts, values='count', names='sentiment', 
                 title='Sentiment Distribution', color_discrete_sequence=['green', 'blue', 'orange', 'red'])
    fig.show()

if __name__ == "__main__":
    visualize_sentiment_distribution()
