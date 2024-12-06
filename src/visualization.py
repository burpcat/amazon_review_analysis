"""
This module implements comprehensive visualization functionality for Amazon product reviews analysis.
It creates visualizations using processed data, topic modeling results, sentiment analysis results,
and helpfulness prediction insights.
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
import warnings
warnings.filterwarnings('ignore')

class ReviewVisualizer:
    def __init__(self, save_dir='visualizations'):
        """Initialize the visualization module."""
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plt.style.use('default')
        sns.set_theme()
        self.color_palette = sns.color_palette("husl", 8)

    def load_processed_data(self):
        """Load the most recent processed data."""
        processed_files = glob.glob('processed_data/processed_reviews_*.csv')
        
        if not processed_files:
            raise FileNotFoundError("Processed data files not found.")
            
        latest_processed = max(processed_files)
        df = pd.read_csv(latest_processed)
        df['Time'] = pd.to_datetime(df['Time'])
        
        print(f"Loaded processed data from: {latest_processed}")
        return df

    def load_analysis_results(self):
        """Load results from all analysis modules."""
        results = {}
        
        # Topic modeling results
        topic_files = glob.glob('topic_models/topic_distributions_*.csv')
        coherence_files = glob.glob('topic_models/coherence_scores_*.csv')
        if topic_files:
            results['topic_distributions'] = pd.read_csv(max(topic_files))
        if coherence_files:
            results['coherence_scores'] = pd.read_csv(max(coherence_files))
        
        # Sentiment analysis results
        sentiment_files = glob.glob('sentiment_analysis/temporal_trends_*.csv')
        if sentiment_files:
            results['temporal_trends'] = pd.read_csv(max(sentiment_files))
        
        # Helpfulness analysis results
        helpfulness_files = glob.glob('helpfulness_analysis/feature_importance_*.csv')
        category_files = glob.glob('helpfulness_analysis/category_patterns_*.csv')
        if helpfulness_files:
            results['feature_importance'] = pd.read_csv(max(helpfulness_files))
        if category_files:
            results['category_patterns'] = pd.read_csv(max(category_files))
        
        return results

    def save_plot(self, fig, filename):
        """Save a matplotlib figure."""
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
        df['month'] = df['Time'].dt.to_period('M')
        df['year'] = df['Time'].dt.year
        
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

    def plot_topic_coherence(self, coherence_scores):
        """Plot topic coherence scores."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=coherence_scores,
            x='n_topics',
            y='coherence_score',
            marker='o'
        )
        ax.set_title('Topic Coherence Scores')
        ax.set_xlabel('Number of Topics')
        ax.set_ylabel('Coherence Score')
        
        self.save_plot(fig, 'topic_coherence')
        return fig

    def plot_feature_importance(self, feature_importance):
        """Plot feature importance for helpfulness prediction."""
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=feature_importance.head(10),
            x='importance',
            y='feature'
        )
        ax.set_title('Top 10 Features for Review Helpfulness')
        
        self.save_plot(fig, 'feature_importance')
        return fig

    def create_interactive_timeline(self, df):
        """Create interactive timeline of review metrics."""
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Average Rating', 'Average Sentiment'))
        
        daily_avg = df.groupby(df['Time'].dt.date).agg({
            'Score': 'mean',
            'sentiment_score': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(x=daily_avg['Time'], y=daily_avg['Score'],
                      mode='lines', name='Average Rating'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=daily_avg['Time'], y=daily_avg['sentiment_score'],
                      mode='lines', name='Average Sentiment'),
            row=2, col=1
        )
        
        fig.update_layout(height=800, title_text="Review Metrics Over Time")
        fig.write_html(f"{self.save_dir}/interactive_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        return fig

    def create_topic_visualization(self, topic_distributions, df):
        """Create interactive topic visualization."""
        topic_sentiment = pd.DataFrame()
        for topic in topic_distributions.columns:
            topic_sentiment[topic] = df['sentiment_score'] * topic_distributions[topic]
        
        topic_summary = pd.DataFrame({
            'topic': topic_distributions.columns,
            'avg_sentiment': topic_sentiment.mean(),
            'prevalence': topic_distributions.mean()
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=topic_summary['prevalence'],
            y=topic_summary['avg_sentiment'],
            mode='markers+text',
            text=topic_summary['topic'],
            textposition="top center",
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Topic Prevalence vs Average Sentiment",
            xaxis_title="Topic Prevalence",
            yaxis_title="Average Sentiment"
        )
        
        fig.write_html(f"{self.save_dir}/topic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        return fig

    def plot_word_cloud(self, df):
        """Create word cloud from review text."""
        try:
            from wordcloud import WordCloud
            text = ' '.join(df['Text'].astype(str))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Word Cloud of Reviews')
            
            self.save_plot(fig, 'word_cloud')
            return fig
        except ImportError:
            print("WordCloud package not installed. Skipping word cloud visualization.")
            return None

    def create_all_visualizations(self):
        """Create all available visualizations."""
        try:
            # Load all data
            print("Loading data and analysis results...")
            df = self.load_processed_data()
            results = self.load_analysis_results()
            
            # Basic visualizations
            print("Creating basic review visualizations...")
            self.plot_rating_distribution(df)
            self.plot_word_cloud(df)
            
            # Sentiment visualizations
            if 'sentiment_score' in df.columns:
                print("Creating sentiment visualizations...")
                self.plot_sentiment_heatmap(df)
                self.create_interactive_timeline(df)
            
            # Helpfulness visualizations
            if 'helpfulness_ratio' in df.columns:
                print("Creating helpfulness analysis...")
                self.plot_helpfulness_analysis(df)
            
            # Topic modeling visualizations
            if 'coherence_scores' in results:
                print("Creating topic modeling visualizations...")
                self.plot_topic_coherence(results['coherence_scores'])
            
            if 'topic_distributions' in results:
                print("Creating topic analysis visualization...")
                self.create_topic_visualization(results['topic_distributions'], df)
            
            # Helpfulness prediction visualizations
            if 'feature_importance' in results:
                print("Creating helpfulness prediction visualizations...")
                self.plot_feature_importance(results['feature_importance'])
            
            print("\nVisualization process completed successfully!")
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise e

def main():
    """Main function to create all visualizations."""
    visualizer = ReviewVisualizer()
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main()