# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from textblob import TextBlob
# import re
# from datetime import datetime
# import os

# def load_and_clean_data(filename='Reviews.csv'):
#     """
#     Load and perform initial cleaning of the Amazon reviews dataset
#     """
#     # Read the data from the same directory
#     df = pd.read_csv(filename)
    
#     # Convert timestamps to datetime
#     df['Time'] = pd.to_datetime(df['Time'], unit='s')
    
#     # Calculate helpfulness ratio with handling division by zero
#     df['helpfulness_ratio'] = np.where(
#         df['HelpfulnessDenominator'] > 0,
#         df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'],
#         0
#     )
    
#     # Convert float NaN to string empty
#     df['Text'] = df['Text'].fillna('').astype(str)
#     df['Summary'] = df['Summary'].fillna('').astype(str)
    
#     # Basic text cleaning
#     df['clean_text'] = df['Text'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
#     df['clean_summary'] = df['Summary'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
    
#     return df

# def perform_sentiment_analysis(df):
#     """
#     Add sentiment analysis scores to the dataset
#     """
#     # Calculate sentiment scores
#     df['sentiment_score'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
#     df['subjectivity_score'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
#     return df

# def create_text_features(df):
#     """
#     Create features from review text
#     """
#     # Already converted to string in load_and_clean_data
#     df['text_length'] = df['Text'].str.len()
#     df['summary_length'] = df['Summary'].str.len()
#     df['word_count'] = df['Text'].str.split().str.len()
    
#     return df

# def prepare_for_topic_modeling(df):
#     """
#     Prepare text data for LDA topic modeling
#     """
#     # Create TF-IDF features
#     tfidf = TfidfVectorizer(max_features=1000,
#                            stop_words='english')
    
#     text_features = tfidf.fit_transform(df['clean_text'])
#     feature_names = tfidf.get_feature_names_out()
    
#     return text_features, feature_names

# def save_processed_data(df, text_features, feature_names):
#     """
#     Save all processed data to files
#     """
#     # Create 'processed_data' directory if it doesn't exist
#     if not os.path.exists('processed_data'):
#         os.makedirs('processed_data')
    
#     # Generate timestamp for unique filenames
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
#     # Save main processed dataframe
#     df.to_csv(f'processed_data/processed_reviews_{timestamp}.csv', index=False)
    
#     # Save text features as sparse matrix
#     import scipy.sparse as sparse
#     sparse.save_npz(f'processed_data/text_features_{timestamp}.npz', text_features)
    
#     # Save feature names
#     pd.Series(feature_names).to_csv(f'processed_data/feature_names_{timestamp}.csv', index=False)
    
#     print(f"Data saved in 'processed_data' directory with timestamp {timestamp}")
#     return timestamp

# def main_processing_pipeline(input_filename='Reviews.csv'):
#     """
#     Main processing pipeline that combines all steps
#     """
#     print("Starting data processing pipeline...")
    
#     # Load and clean data
#     print("Loading and cleaning data...")
#     df = load_and_clean_data(input_filename)
    
#     # Add sentiment analysis
#     print("Performing sentiment analysis...")
#     df = perform_sentiment_analysis(df)
    
#     # Create text features
#     print("Creating text features...")
#     df = create_text_features(df)
    
#     # Prepare for topic modeling
#     print("Preparing text for topic modeling...")
#     text_features, feature_names = prepare_for_topic_modeling(df)
    
#     # Save all processed data
#     print("Saving processed data...")
#     timestamp = save_processed_data(df, text_features, feature_names)
    
#     print("Processing complete!")
#     return df, text_features, feature_names, timestamp

# if __name__ == "__main__":
#     # Run the pipeline
#     df, text_features, feature_names, timestamp = main_processing_pipeline()

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import re
from datetime import datetime
import os
from product_category_classifier import ProductCategoryClassifier
from tqdm.auto import tqdm
tqdm.pandas()

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

def classify_product_categories(df, confidence_threshold=0.6):
    """
    Classify products into categories with confidence threshold
    """
    print("Initializing product category classifier...")
    classifier = ProductCategoryClassifier(confidence_threshold=confidence_threshold)
    
    print("Training classifier...")
    classifier.train()
    
    print("Categorizing reviews...")
    df = classifier.categorize_reviews(df)
    
    # Print category distribution
    print("\nCategory Distribution:")
    print(df['predicted_category'].value_counts())
    print("\nConfidence Statistics:")
    print(df['category_confidence'].describe())
    
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
    
    # Get categories excluding 'Uncategorized'
    categories = df['predicted_category'].unique()
    categories = categories[categories != 'Uncategorized']
    
    category_features = {}
    category_feature_names = {}
    
    print("Processing features for each category...")
    for category in tqdm(categories, desc="Categories"):
        # Filter reviews for this category
        category_df = df[df['predicted_category'] == category]
        
        if len(category_df) > 0:
            # Create features for this category
            text_features = tfidf.fit_transform(category_df['clean_text'])
            feature_names = tfidf.get_feature_names_out()
            
            category_features[category] = text_features
            category_feature_names[category] = feature_names
            
            print(f"\nProcessed {category}: {len(category_df)} reviews")
    
    return category_features, category_feature_names

def save_processed_data(df, category_features, category_feature_names):
    """
    Save all processed data to files
    """
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("Saving main dataset...")
    df.to_csv(f'processed_data/processed_reviews_{timestamp}.csv', index=False)
    
    print("Saving category-specific features...")
    for category in tqdm(category_features.keys(), desc="Saving category features"):
        # Skip uncategorized
        if category == 'Uncategorized':
            continue
            
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

def main_processing_pipeline(input_filename='Reviews.csv', confidence_threshold=0.6):
    """
    Main processing pipeline that combines all steps
    """
    print("\n=== Starting data processing pipeline ===\n")
    
    print("Step 1/6: Loading and cleaning data...")
    df = load_and_clean_data(input_filename)
    
    print("\nStep 2/6: Classifying products into categories...")
    df = classify_product_categories(df, confidence_threshold)
    
    print("\nStep 3/6: Performing sentiment analysis...")
    df = perform_sentiment_analysis(df)
    
    print("\nStep 4/6: Creating text features...")
    df = create_text_features(df)
    
    print("\nStep 5/6: Preparing text for topic modeling...")
    category_features, category_feature_names = prepare_for_topic_modeling(df)
    
    print("\nStep 6/6: Saving processed data...")
    timestamp = save_processed_data(df, category_features, category_feature_names)
    
    print("\n=== Processing complete! ===")
    
    # Print final statistics
    print("\nFinal Category Distribution (excluding Uncategorized):")
    categorized_df = df[df['predicted_category'] != 'Uncategorized']
    print(categorized_df['predicted_category'].value_counts())
    
    return df, category_features, category_feature_names, timestamp

if __name__ == "__main__":
    # Run the pipeline with confidence threshold
    df, category_features, category_feature_names, timestamp = main_processing_pipeline(
        confidence_threshold=0.6
    )