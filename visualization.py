"""
visualization.py

This module implements comprehensive visualization functionality for Amazon product reviews analysis.
It creates visualizations using processed data, topic modeling results, and sentiment analysis results.

Dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- wordcloud
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import glob
import scipy.sparse as sparse

class ReviewVisualizer:
    def __init__(self, save_dir='visualizations'):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plt.style.use('default')
        sns.set_theme()
        self.color_palette = sns.color_palette("husl", 8)

    def load_processed_data(self):
        processed_files = glob.glob('processed_data/processed_reviews_*.csv')
        
        if not processed_files:
            raise FileNotFoundError("Processed data files not found. Run amazon_review_processor.py first.")
            
        latest_processed = max(processed_files)
        df = pd.read_csv(latest_processed)
        df['Time'] = pd.to_datetime(df['Time'])
        
        print(f"Loaded processed data from: {latest_processed}")
        return df, None, None

    def save_plot(self, fig, filename):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f"{self.save_dir}/{filename}_{timestamp}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {filepath}")

    def plot_rating_distribution(self, df):
        """Create histogram of review ratings."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='Score', discrete=True, ax=ax)
        ax.set_title('Distribution of Review Ratings')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        
        self.save_plot(fig, 'rating_distribution')
        return fig

    def plot_sentiment_heatmap(self, df):
        """Create heatmap of sentiment scores over time."""
        # Convert Time to datetime if it's not already
        df['Time'] = pd.to_datetime(df['Time'])
        df['month'] = df['Time'].dt.to_period('M')
        df['year'] = df['Time'].dt.year
        
        # Create pivot table
        pivot_table = df.pivot_table(
            values='sentiment_score',
            index='month',
            columns='year',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_table, cmap='RdYlBu', center=0, ax=ax)
        ax.set_title('Sentiment Scores Over Time')
        
        self.save_plot(fig, 'sentiment_heatmap')
        return fig

    def plot_helpfulness_analysis(self, df):
        """Create scatter plot of helpfulness ratio vs. review length."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x='text_length',
            y='helpfulness_ratio',
            alpha=0.5,
            ax=ax
        )
        ax.set_title('Review Helpfulness vs. Length')
        ax.set_xlabel('Review Length (characters)')
        ax.set_ylabel('Helpfulness Ratio')
        
        self.save_plot(fig, 'helpfulness_analysis')
        return fig

    def create_interactive_timeline(self, df):
        """Create interactive timeline of review metrics."""
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Average Rating', 'Average Sentiment'))
        
        # Calculate daily averages
        daily_avg = df.groupby(df['Time'].dt.date).agg({
            'Score': 'mean',
            'sentiment_score': 'mean'
        }).reset_index()
        
        # Add rating timeline
        fig.add_trace(
            go.Scatter(x=daily_avg['Time'], y=daily_avg['Score'],
                      mode='lines', name='Average Rating'),
            row=1, col=1
        )
        
        # Add sentiment timeline
        fig.add_trace(
            go.Scatter(x=daily_avg['Time'], y=daily_avg['sentiment_score'],
                      mode='lines', name='Average Sentiment'),
            row=2, col=1
        )
        
        fig.update_layout(height=800, title_text="Review Metrics Over Time")
        fig.write_html(f"{self.save_dir}/interactive_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        return fig

    def plot_word_cloud(self, df):
        """Create word cloud from review text."""
        try:
            from wordcloud import WordCloud
            
            # Combine all text
            text = ' '.join(df['Text'].astype(str))
            
            # Create word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Word Cloud of Reviews')
            
            self.save_plot(fig, 'word_cloud')
            return fig
        except ImportError:
            print("WordCloud package not installed. Skipping word cloud visualization.")
            return None

    def create_all_visualizations(self, df):
        """Create all available visualizations."""
        print("Creating basic review visualizations...")
        self.plot_rating_distribution(df)
        
        if 'sentiment_score' in df.columns:
            print("Creating sentiment visualizations...")
            self.plot_sentiment_heatmap(df)
            self.create_interactive_timeline(df)
            
        if 'helpfulness_ratio' in df.columns and 'text_length' in df.columns:
            print("Creating helpfulness analysis...")
            self.plot_helpfulness_analysis(df)
            
        print("Creating word cloud...")
        self.plot_word_cloud(df)

def main():
    """Main function to create all visualizations."""
    try:
        # Initialize visualizer
        visualizer = ReviewVisualizer()
        
        # Load processed data
        print("Loading data...")
        df, _, _ = visualizer.load_processed_data()
        
        # Create all visualizations
        visualizer.create_all_visualizations(df)
        
        print("\nVisualization process completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()