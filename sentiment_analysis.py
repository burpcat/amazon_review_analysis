"""
sentiment_analysis.py

This module implements advanced sentiment analysis functionality for Amazon product reviews.
Loads preprocessed data and performs detailed sentiment analysis including aspect-based analysis.

Dependencies:
- pandas
- numpy
- textblob
- nltk
- scikit-learn
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from datetime import datetime
import glob

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the SentimentAnalyzer with default parameters."""
        self.aspect_lexicon = {
            'quality': ['quality', 'durability', 'durable', 'sturdy', 'solid'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value'],
            'usability': ['easy', 'simple', 'difficult', 'complicated', 'user-friendly'],
            'service': ['delivery', 'shipping', 'customer service', 'support', 'warranty']
        }
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
    def load_processed_data(self):
        """Load the most recent processed review data."""
        processed_files = glob.glob('processed_data/processed_reviews_*.csv')
        
        if not processed_files:
            raise FileNotFoundError("Processed data files not found. Run amazon_review_processor.py first.")
            
        latest_file = max(processed_files)
        df = pd.read_csv(latest_file)
        print(f"Loaded processed data from: {latest_file}")
        
        return df
        
    def extract_aspect_sentiments(self, text):
        """
        Extract sentiment scores for different aspects of a review using simple splitting.
        """
        # Split text into sentences using simple period-based splitting
        sentences = str(text).split('.')
        aspect_sentiments = {aspect: [] for aspect in self.aspect_lexicon}
        
        for sentence in sentences:
            if sentence.strip():  # Check if sentence is not empty
                blob = TextBlob(sentence.lower())
                sentence_sentiment = blob.sentiment.polarity
                
                # Check each aspect
                for aspect, terms in self.aspect_lexicon.items():
                    if any(term in sentence for term in terms):
                        aspect_sentiments[aspect].append(sentence_sentiment)
        
        # Average sentiments for each aspect
        return {
            aspect: np.mean(sentiments) if sentiments else np.nan
            for aspect, sentiments in aspect_sentiments.items()
        }
    
    def train_sentiment_classifier(self, texts, ratings):
        """Train a sentiment classifier using review texts and ratings."""
        # Convert ratings to sentiment labels
        labels = pd.cut(ratings, bins=[-np.inf, 2, 3, np.inf], labels=['negative', 'neutral', 'positive'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        print("Training sentiment classifier...")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_vec, y_train)
        
        accuracy = self.model.score(X_test_vec, y_test)
        print(f"Training completed with accuracy: {accuracy:.4f}")
        
        return accuracy

    def save_results(self, aspect_sentiments_df, trends, directory='sentiment_analysis'):
        """Save sentiment analysis results."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save aspect sentiments
        aspect_sentiments_df.to_csv(f'{directory}/aspect_sentiments_{timestamp}.csv', index=False)
        
        # Save trends
        pd.DataFrame(trends['time_trends']).to_csv(f'{directory}/time_trends_{timestamp}.csv')
        pd.DataFrame(trends['rating_trends']).to_csv(f'{directory}/rating_trends_{timestamp}.csv')
        
        # Save models
        if self.model is not None:
            with open(f'{directory}/sentiment_classifier_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            with open(f'{directory}/sentiment_vectorizer_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
        
        print(f"Results and models saved in {directory} directory with timestamp {timestamp}")

def analyze_sentiment_trends(df):
    """Analyze sentiment trends over time and by product category."""
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Analyze trends over time
    time_trends = df.groupby(df['Time'].dt.to_period('M'))['sentiment_score'].mean()
    
    # Analyze trends by rating
    rating_trends = df.groupby('Score')['sentiment_score'].mean()
    
    return {
        'time_trends': time_trends,
        'rating_trends': rating_trends
    }

def main():
    """Main function to run sentiment analysis on processed review data."""
    try:
        # Initialize analyzer
        analyzer = SentimentAnalyzer()
        
        # Load processed data
        df = analyzer.load_processed_data()
        
        print("Starting sentiment analysis...")
        
        # Train sentiment classifier
        accuracy = analyzer.train_sentiment_classifier(df['Text'], df['Score'])
        
        # Extract aspect sentiments for a sample of reviews
        print("\nAnalyzing aspect-based sentiments...")
        sample_size = min(1000, len(df))  # Analyze up to 1000 reviews
        sample_reviews = df['Text'].head(sample_size)
        aspect_sentiments = [analyzer.extract_aspect_sentiments(text) for text in sample_reviews]
        aspect_df = pd.DataFrame(aspect_sentiments)
        
        print("\nAverage aspect sentiments:")
        print(aspect_df.mean())
        
        # Analyze trends
        print("\nAnalyzing sentiment trends...")
        trends = analyze_sentiment_trends(df)
        
        # Save all results
        analyzer.save_results(aspect_df, trends)
        
        print("\nSentiment analysis completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()