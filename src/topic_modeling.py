"""
Enhanced version that performs category-specific topic modeling with coherence optimization
and sentiment analysis.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import pickle
import os
from datetime import datetime
import scipy.sparse as sparse
import glob
from sklearn.feature_extraction.text import CountVectorizer
import warnings
from tqdm.auto import tqdm
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore', message=".*OpenSSL.*")
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

os.environ["LOKY_MAX_CPU_COUNT"] = "1"  # Limit parallel processing

class TopicModeler:
    def __init__(self, min_topics=5, max_topics=15, max_iter=20, random_state=42):
        """Initialize the TopicModeler with given parameters."""
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.max_iter = max_iter
        self.random_state = random_state
        self.vectorizer = CountVectorizer(max_features=1000)
        self.category_models = {}
        self.category_coherence_scores = {}
        
    def load_processed_data(self):
        """Load the most recently processed data files."""
        print("Loading processed data...")
        
        processed_files = glob.glob('processed_data/processed_reviews_*.csv')
        if not processed_files:
            raise FileNotFoundError("Processed data files not found. Run amazon_review_processor.py first.")
            
        latest_processed = max(processed_files)
        df = pd.read_csv(latest_processed)
        
        # Get unique categories
        categories = df['predicted_category'].unique()
        
        # Load category-specific features
        category_features = {}
        category_feature_names = {}
        
        print("Loading category-specific features...")
        for category in tqdm(categories, desc="Loading categories"):
            category_dir = f'processed_data/{category.lower().replace(" & ", "_")}'
            feature_files = glob.glob(f'{category_dir}/text_features_*.npz')
            names_files = glob.glob(f'{category_dir}/feature_names_*.csv')
            
            if feature_files and names_files:
                latest_features = max(feature_files)
                latest_names = max(names_files)
                
                category_features[category] = sparse.load_npz(latest_features)
                category_feature_names[category] = pd.read_csv(latest_names).iloc[:, 0].tolist()
        
        return df, category_features, category_feature_names

    def compute_coherence_values(self, texts, dictionary, corpus, step=5, max_docs=10000):
        """Compute coherence scores for different numbers of topics."""
        coherence_scores = {}
        
        # Limit the number of documents for coherence computation
        if len(texts) > max_docs:
            print(f"Sampling {max_docs} documents for coherence computation...")
            indices = np.random.choice(len(texts), max_docs, replace=False)
            texts = [texts[i] for i in indices]
            corpus = [corpus[i] for i in indices]
        
        for num_topics in tqdm(range(self.min_topics, self.max_topics + 1, step), 
                            desc="Computing coherence scores"):
            print(f"\nTesting {num_topics} topics...")
            lda_model = LdaModel(
                corpus=corpus,
                num_topics=num_topics,
                id2word=dictionary,
                random_state=self.random_state,
                iterations=50  # Reduced iterations for faster computation
            )
            
            try:
                coherence_model = CoherenceModel(
                    model=lda_model,
                    texts=texts,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                
                coherence_score = coherence_model.get_coherence()
                coherence_scores[num_topics] = coherence_score
                print(f"Coherence score for {num_topics} topics: {coherence_score}")
            except Exception as e:
                print(f"Error computing coherence for {num_topics} topics: {str(e)}")
                coherence_scores[num_topics] = 0
        
        return coherence_scores
    
    """
        Below function uses entire dataset for calculating coherence rather than only using 10000 docs
    """
    # def compute_coherence_values(self, texts, dictionary, corpus, step=5, max_docs=None):
    #     """Compute coherence scores for different numbers of topics."""
    #     coherence_scores = {}
        
    #     # Option to use full dataset
    #     if max_docs:
    #         print(f"Sampling {max_docs} documents for coherence computation...")
    #         indices = np.random.choice(len(texts), max_docs, replace=False)
    #         texts = [texts[i] for i in indices]
    #         corpus = [corpus[i] for i in indices]
    #     else:
    #         print(f"Using full dataset ({len(texts)} documents) for coherence computation...")
        
    #     for num_topics in tqdm(range(self.min_topics, self.max_topics + 1, step), 
    #                         desc="Computing coherence scores"):
    #         print(f"\nTesting {num_topics} topics...")
    #         lda_model = LdaModel(
    #             corpus=corpus,
    #             num_topics=num_topics,
    #             id2word=dictionary,
    #             random_state=self.random_state,
    #             iterations=50
    #         )
            
    #         try:
    #             coherence_model = CoherenceModel(
    #                 model=lda_model,
    #                 texts=texts,
    #                 dictionary=dictionary,
    #                 coherence='c_v'
    #             )
                
    #             coherence_score = coherence_model.get_coherence()
    #             coherence_scores[num_topics] = coherence_score
    #             print(f"Coherence score for {num_topics} topics: {coherence_score}")
    #         except Exception as e:
    #             print(f"Error computing coherence for {num_topics} topics: {str(e)}")
    #             coherence_scores[num_topics] = 0
        
    #     return coherence_scores

    def find_optimal_topics(self, df, category):
        """Find the optimal number of topics for a category using coherence scores."""
        print(f"\nFinding optimal number of topics for {category}...")
        
        # Filter texts for this category
        category_texts = df[df['predicted_category'] == category]['clean_text']
        
        # Prepare texts for coherence calculation
        texts = [text.split() for text in category_texts]
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        # Compute coherence scores
        coherence_scores = self.compute_coherence_values(texts, dictionary, corpus)
        self.category_coherence_scores[category] = coherence_scores
        
        # Find optimal number of topics
        optimal_topics = max(coherence_scores.items(), key=lambda x: x[1])[0]
        print(f"Optimal number of topics for {category}: {optimal_topics}")
        
        return optimal_topics

    def fit_category(self, category, text_features, optimal_topics):
            """Fit LDA model for a specific category with improved handling for large datasets."""
            print(f"\nFitting LDA model for {category} with {optimal_topics} topics...")
            print(f"Total documents in category: {text_features.shape[0]}")
            
            # Configure the model for efficient processing of large datasets
            lda_model = LatentDirichletAllocation(
                n_components=optimal_topics,
                max_iter=self.max_iter,
                random_state=self.random_state,
                batch_size=4096,  # Increased batch size for better performance
                n_jobs=1,  # Avoid multiprocessing issues
                learning_method='online',  # Use online learning for better memory efficiency
                learning_offset=50.0,  # Slower learning rate
                evaluate_every=5  # Reduce frequency of perplexity evaluation
            )
            
            # Fit the model with chunking for large datasets
            chunk_size = 10000
            n_samples = text_features.shape[0]
            n_chunks = (n_samples + chunk_size - 1) // chunk_size
            
            print(f"Processing in {n_chunks} chunks...")
            
            for i in tqdm(range(n_chunks), desc="Processing chunks"):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_samples)
                chunk = text_features[start_idx:end_idx]
                
                if i == 0:
                    # First chunk: fit
                    document_topics = lda_model.fit_transform(chunk)
                else:
                    # Subsequent chunks: partial_fit
                    chunk_topics = lda_model.transform(chunk)
                    document_topics = np.vstack([document_topics, chunk_topics])
            
            print("Model fitting completed!")
            
            self.category_models[category] = {
                'model': lda_model,
                'document_topics': document_topics,
                'optimal_topics': optimal_topics
            }
            
            return document_topics

    def get_top_terms_per_topic(self, category, feature_names, n_terms=10):
        """Extract the top terms for a category, combining all topics."""
        if category not in self.category_models:
            return {}
            
        model = self.category_models[category]['model']
        all_terms = []
        
        # Combine terms from all topics
        for topic_idx, topic in enumerate(model.components_):
            top_terms_idx = topic.argsort()[:-n_terms-1:-1]
            top_terms = [feature_names[i] for i in top_terms_idx]
            all_terms.extend(top_terms)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in all_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return {category: unique_terms}

    def analyze_category_sentiment(self, df, category):
        """Analyze sentiment distribution across topics for a category."""
        if 'sentiment_score' not in df.columns:
            print(f"Warning: No sentiment scores found for {category}")
            return None
            
        category_df = df[df['predicted_category'] == category]
        document_topics = self.category_models[category]['document_topics']
        optimal_topics = self.category_models[category]['optimal_topics']
        
        topic_sentiment_analysis = pd.DataFrame({
            'topic': range(optimal_topics),
            'avg_sentiment': [
                np.average(category_df['sentiment_score'], weights=document_topics[:, i])
                for i in range(optimal_topics)
            ]
        })
        
        return topic_sentiment_analysis

    def save_results(self, df, category_features, category_feature_names):
        """Save topic modeling results and visualizations for all categories."""
        if not os.path.exists('topic_models'):
            os.makedirs('topic_models')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print("\nSaving results for each category...")
        for category in tqdm(self.category_models.keys(), desc="Saving results"):
            # Create category directory
            category_dir = f'topic_models/{category.lower().replace(" & ", "_")}'
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
            
            # Save document-topic distributions
            topic_distributions = pd.DataFrame(
                self.category_models[category]['document_topics'],
                columns=[f'Topic_{i+1}' for i in range(self.category_models[category]['optimal_topics'])]
            )
            topic_distributions.to_csv(f'{category_dir}/topic_distributions_{timestamp}.csv', index=False)
            
            # Save top terms for the category
            top_terms = self.get_top_terms_per_topic(category, category_feature_names[category])
            with open(f'{category_dir}/top_terms_{timestamp}.txt', 'w') as f:
                f.write(f"Terms for {category}:\n")
                terms = top_terms[category]
                f.write(", ".join(terms) + "\n\n")

def main():
    """Main function to run enhanced topic modeling."""
    try:
        print("Starting enhanced topic modeling process...")
        
        # Initialize topic modeler
        modeler = TopicModeler(min_topics=5, max_topics=15)
        
        # Load processed data
        df, category_features, category_feature_names = modeler.load_processed_data()
        
        # Process each category
        for category in category_features.keys():
            # Find optimal number of topics
            optimal_topics = modeler.find_optimal_topics(df, category)
            
            # Fit model for category
            document_topics = modeler.fit_category(
                category, 
                category_features[category], 
                optimal_topics
            )
            
            # Print top terms
            print(f"\nTerms for {category}:")
            top_terms = modeler.get_top_terms_per_topic(
                category, 
                category_feature_names[category]
            )
            for category_name, terms in top_terms.items():
                print(", ".join(terms))
        
        # Save all results
        modeler.save_results(df, category_features, category_feature_names)
        
        print("\nEnhanced topic modeling completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()