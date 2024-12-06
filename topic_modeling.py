"""
topic_modeling.py

This module implements topic modeling functionality for Amazon product reviews using
Latent Dirichlet Allocation (LDA). It loads preprocessed data and generates topic insights.

Dependencies:
- numpy
- pandas
- sklearn
- gensim
- pyLDAvis
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import pickle
import os
from datetime import datetime
import scipy.sparse as sparse
import glob

class TopicModeler:
    def __init__(self, n_topics=10, max_iter=20, random_state=42):
        """Initialize the TopicModeler with given parameters."""
        self.n_topics = n_topics
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            random_state=random_state
        )
        
    def load_processed_data(self):
        """
        Load the most recently processed data files.
        
        Returns:
            tuple: (processed_df, text_features, feature_names)
        """
        # Find most recent files
        processed_files = glob.glob('processed_data/processed_reviews_*.csv')
        feature_files = glob.glob('processed_data/text_features_*.npz')
        names_files = glob.glob('processed_data/feature_names_*.csv')
        
        if not (processed_files and feature_files and names_files):
            raise FileNotFoundError("Processed data files not found. Run amazon_review_processor.py first.")
            
        # Get most recent files
        latest_processed = max(processed_files)
        latest_features = max(feature_files)
        latest_names = max(names_files)
        
        # Load the data
        df = pd.read_csv(latest_processed)
        text_features = sparse.load_npz(latest_features)
        feature_names = pd.read_csv(latest_names).iloc[:, 0].tolist()
        
        print(f"Loaded processed data from: {latest_processed}")
        return df, text_features, feature_names
    
    def fit(self, text_features):
        """
        Fit the LDA model to the text features.
        
        Args:
            text_features: Sparse matrix of text features
            
        Returns:
            document_topics: Document-topic matrix
        """
        print("Fitting LDA model...")
        self.document_topics = self.lda_model.fit_transform(text_features)
        return self.document_topics
    
    def get_top_terms_per_topic(self, feature_names, n_terms=10):
        """
        Extract the top terms for each topic.
        
        Args:
            feature_names: List of feature names
            n_terms: Number of terms to extract per topic
            
        Returns:
            dict: Dictionary mapping topic IDs to lists of top terms
        """
        topics = {}
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_terms_idx = topic.argsort()[:-n_terms-1:-1]
            top_terms = [feature_names[i] for i in top_terms_idx]
            topics[f"Topic {topic_idx+1}"] = top_terms
        return topics
    
    def save_results(self, df, text_features, feature_names):
        """
        Save topic modeling results and visualizations.
        
        Args:
            df: Processed DataFrame
            text_features: Text feature matrix
            feature_names: List of feature names
        """
        # Create directories if they don't exist
        if not os.path.exists('topic_models'):
            os.makedirs('topic_models')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save document-topic distributions
        topic_distributions = pd.DataFrame(
            self.document_topics,
            columns=[f'Topic_{i+1}' for i in range(self.n_topics)]
        )
        topic_distributions.to_csv(f'topic_models/topic_distributions_{timestamp}.csv', index=False)
        
        # Save top terms for each topic
        top_terms = self.get_top_terms_per_topic(feature_names)
        with open(f'topic_models/top_terms_{timestamp}.txt', 'w') as f:
            for topic, terms in top_terms.items():
                f.write(f"{topic}:\n{', '.join(terms)}\n\n")
        
        # Create and save pyLDAvis visualization
        vis_data = pyLDAvis.sklearn.prepare(
            self.lda_model,
            text_features,
            self.vectorizer,
            mds='tsne'
        )
        pyLDAvis.save_html(vis_data, f'topic_models/topic_visualization_{timestamp}.html')
        
        print(f"Results saved in topic_models directory with timestamp {timestamp}")

def main():
    """
    Main function to run topic modeling on processed review data.
    """
    try:
        # Initialize topic modeler
        modeler = TopicModeler()
        
        # Load processed data
        df, text_features, feature_names = modeler.load_processed_data()
        
        # Fit model and get document topics
        document_topics = modeler.fit(text_features)
        
        # Get and print top terms for each topic
        top_terms = modeler.get_top_terms_per_topic(feature_names)
        print("\nTop terms per topic:")
        for topic, terms in top_terms.items():
            print(f"\n{topic}:")
            print(", ".join(terms))
        
        # Save results
        modeler.save_results(df, text_features, feature_names)
        
        print("\nTopic modeling completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

"""
category_keyword_analysis.py

This module analyzes and extracts key terms for each product category
from Amazon reviews.
"""

