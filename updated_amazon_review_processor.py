import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import re
from datetime import datetime
import os
from product_category_classifier import ProductCategoryClassifier
from tqdm.auto import tqdm  # Added for progress bars
tqdm.pandas()  # Enable progress_apply for pandas

def load_and_clean_data(filename='Reviews.csv'):
    """
    Load and perform initial cleaning of the Amazon reviews dataset
    """
    print("Loading data...")
    df = pd.read_csv(filename)
    
    print("Converting timestamps...")
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    
    print("Calculating helpfulness ratios...")
    df['helpfulness_ratio'] = np.where(
        df['HelpfulnessDenominator'] > 0,
        df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'],
        0
    )
    
    print("Cleaning text fields...")
    df['Text'] = df['Text'].fillna('').astype(str)
    df['Summary'] = df['Summary'].fillna('').astype(str)
    
    print("Performing text preprocessing...")
    tqdm.pandas(desc="Cleaning review text")
    df['clean_text'] = df['Text'].progress_apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
    
    tqdm.pandas(desc="Cleaning summary text")
    df['clean_summary'] = df['Summary'].progress_apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
    
    return df

def perform_sentiment_analysis(df):
    """
    Add sentiment analysis scores to the dataset
    """
    print("Calculating sentiment scores...")
    tqdm.pandas(desc="Sentiment analysis")
    df['sentiment_score'] = df['Text'].progress_apply(lambda x: TextBlob(x).sentiment.polarity)
    
    tqdm.pandas(desc="Subjectivity analysis")
    df['subjectivity_score'] = df['Text'].progress_apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
    return df

def create_text_features(df):
    """
    Create features from review text
    """
    print("Creating text features...")
    df['text_length'] = df['Text'].str.len()
    df['summary_length'] = df['Summary'].str.len()
    df['word_count'] = df['Text'].str.split().str.len()
    
    return df

def prepare_for_topic_modeling(df):
    """
    Prepare text data for LDA topic modeling - now with category-specific processing
    """
    tfidf = TfidfVectorizer(max_features=1000,
                           stop_words='english')
    
    categories = df['predicted_category'].unique()
    category_features = {}
    category_feature_names = {}
    
    print("Processing features for each category...")
    for category in tqdm(categories, desc="Categories"):
        category_df = df[df['predicted_category'] == category]
        
        if len(category_df) > 0:
            text_features = tfidf.fit_transform(category_df['clean_text'])
            feature_names = tfidf.get_feature_names_out()
            
            category_features[category] = text_features
            category_feature_names[category] = feature_names
    
    return category_features, category_feature_names

def classify_product_categories(df):
    """
    Classify products into categories
    """
    print("Initializing product category classifier...")
    classifier = ProductCategoryClassifier()
    
    print("Training classifier...")
    classifier.train()
    
    print("Categorizing reviews...")
    # Combine Summary and Text for better classification
    tqdm.pandas(desc="Preparing texts for classification")
    combined_texts = df['Summary'] + ' ' + df['Text']
    
    print("Converting reviews to TF-IDF features...")
    X = classifier.vectorizer.transform(combined_texts)
    
    print("Predicting categories...")
    categories = classifier.classifier.predict(X)
    probabilities = np.max(classifier.classifier.predict_proba(X), axis=1)
    
    df['predicted_category'] = categories
    df['category_confidence'] = probabilities
    
    return df

def save_processed_data(df, category_features, category_feature_names):
    """
    Save all processed data to files
    """
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("Saving processed data...")
    df.to_csv(f'processed_data/processed_reviews_{timestamp}.csv', index=False)
    
    print("Saving category-specific features...")
    for category in tqdm(category_features.keys(), desc="Saving category features"):
        category_dir = f'processed_data/{category.lower().replace(" & ", "_")}'
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        
        import scipy.sparse as sparse
        sparse.save_npz(
            f'{category_dir}/text_features_{timestamp}.npz',
            category_features[category]
        )
        
        pd.Series(category_feature_names[category]).to_csv(
            f'{category_dir}/feature_names_{timestamp}.csv',
            index=False
        )
    
    print(f"Data saved in 'processed_data' directory with timestamp {timestamp}")
    return timestamp

def main_processing_pipeline(input_filename='Reviews.csv'):
    """
    Main processing pipeline that combines all steps
    """
    print("\n=== Starting data processing pipeline ===\n")
    
    print("Step 1/6: Loading and cleaning data...")
    df = load_and_clean_data(input_filename)
    
    print("\nStep 2/6: Classifying products into categories...")
    df = classify_product_categories(df)
    
    print("\nStep 3/6: Performing sentiment analysis...")
    df = perform_sentiment_analysis(df)
    
    print("\nStep 4/6: Creating text features...")
    df = create_text_features(df)
    
    print("\nStep 5/6: Preparing text for topic modeling...")
    category_features, category_feature_names = prepare_for_topic_modeling(df)
    
    print("\nStep 6/6: Saving processed data...")
    timestamp = save_processed_data(df, category_features, category_feature_names)
    
    print("\n=== Processing complete! ===")
    return df, category_features, category_feature_names, timestamp

if __name__ == "__main__":
    # Run the pipeline
    df, category_features, category_feature_names, timestamp = main_processing_pipeline()
    
    # Print category distribution
    print("\nCategory Distribution:")
    print(df['predicted_category'].value_counts())
    
    # Print sample of categorized reviews
    print("\nSample of Categorized Reviews:")
    sample_reviews = df[['Summary', 'predicted_category', 'category_confidence']].sample(5)
    print(sample_reviews)