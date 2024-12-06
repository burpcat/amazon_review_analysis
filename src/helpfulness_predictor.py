"""
helpfulness_predictor.py

This module implements predictive modeling for review helpfulness analysis.
It includes feature engineering, model training, and analysis of factors
that contribute to review helpfulness.

Dependencies:
- pandas
- numpy
- scikit-learn
- lightgbm
- shap
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import glob

class HelpfulnessPredictor:
    def __init__(self):
        """Initialize the helpfulness predictor."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def load_processed_data(self):
        """Load the most recent processed review data."""
        processed_files = glob.glob('processed_data/processed_reviews_*.csv')
        
        if not processed_files:
            raise FileNotFoundError("Processed data files not found.")
            
        latest_file = max(processed_files)
        df = pd.read_csv(latest_file)
        print(f"Loaded processed data from: {latest_file}")
        
        return df

    def create_features(self, df):
        """
        Create features for helpfulness prediction.
        
        Args:
            df: DataFrame containing review data
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame()
        
        # Basic features
        features['text_length'] = df['text_length']
        features['word_count'] = df['word_count']
        features['avg_word_length'] = df['text_length'] / df['word_count']
        features['rating'] = df['Score']
        
        # Sentiment features
        if 'sentiment_score' in df.columns:
            features['sentiment_score'] = df['sentiment_score']
            features['sentiment_magnitude'] = abs(df['sentiment_score'])
        
        # Rating deviation
        features['rating_deviation'] = abs(df['Score'] - df['Score'].mean())
        
        # Time-based features
        df['Time'] = pd.to_datetime(df['Time'])
        features['hour'] = df['Time'].dt.hour
        features['day_of_week'] = df['Time'].dt.dayofweek
        
        self.feature_names = features.columns.tolist()
        return features

    def prepare_target(self, df):
        """Prepare target variable (helpfulness ratio)."""
        return df['helpfulness_ratio']

    def train_model(self, X, y):
        """
        Train the helpfulness prediction model.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Trained model and evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'test_data': (X_test_scaled, y_test, y_pred)
        }

    def analyze_feature_importance(self, X):
        """
        Analyze feature importance using SHAP values.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with feature importance analysis
        """
        # Calculate SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        # Compute feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values).mean(0),
            'shap_values': list(shap_values.T)
        })
        
        return importance_df.sort_values('importance', ascending=False)

    def analyze_category_patterns(self, df, features, y):
        """
        Analyze helpfulness patterns by product category.
        
        Args:
            df: Original DataFrame
            features: Feature matrix
            y: Target variable
            
        Returns:
            DataFrame with category-specific analysis
        """
        # Use product IDs as categories
        if 'ProductId' in df.columns:
            category_patterns = pd.DataFrame()
            
            for category in df['ProductId'].unique():
                mask = df['ProductId'] == category
                
                if mask.sum() > 50:  # Only analyze categories with sufficient data
                    category_data = {
                        'category': category,
                        'avg_helpfulness': y[mask].mean(),
                        'review_count': mask.sum(),
                        'avg_rating': df.loc[mask, 'Score'].mean()
                    }
                    
                    category_patterns = pd.concat([
                        category_patterns,
                        pd.DataFrame([category_data])
                    ])
            
            return category_patterns
        
        return None

    def save_results(self, results_dict, directory='helpfulness_analysis'):
        """Save analysis results and model."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save feature importance
        if 'feature_importance' in results_dict:
            results_dict['feature_importance'].to_csv(
                f'{directory}/feature_importance_{timestamp}.csv'
            )
        
        # Save category patterns
        if 'category_patterns' in results_dict:
            results_dict['category_patterns'].to_csv(
                f'{directory}/category_patterns_{timestamp}.csv'
            )
        
        # Save model performance metrics
        if 'metrics' in results_dict:
            pd.DataFrame([results_dict['metrics']]).to_csv(
                f'{directory}/model_metrics_{timestamp}.csv'
            )
        
        print(f"Results saved in {directory} directory with timestamp {timestamp}")

    def create_visualizations(self, results_dict, directory='helpfulness_analysis'):
        """Create visualizations of the analysis results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Feature importance plot
        if 'feature_importance' in results_dict:
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=results_dict['feature_importance'].head(10),
                x='importance',
                y='feature'
            )
            plt.title('Top 10 Features for Helpfulness Prediction')
            plt.tight_layout()
            plt.savefig(f'{directory}/feature_importance_{timestamp}.png')
            plt.close()
        
        # Category patterns plot
        if 'category_patterns' in results_dict:
            plt.figure(figsize=(12, 6))
            sns.scatterplot(
                data=results_dict['category_patterns'],
                x='avg_rating',
                y='avg_helpfulness',
                size='review_count',
                alpha=0.6
            )
            plt.title('Helpfulness vs Rating by Category')
            plt.tight_layout()
            plt.savefig(f'{directory}/category_patterns_{timestamp}.png')
            plt.close()

def main():
    """Main function to run helpfulness prediction analysis."""
    try:
        # Initialize predictor
        predictor = HelpfulnessPredictor()
        
        # Load data
        print("Loading data...")
        df = predictor.load_processed_data()
        
        # Create features
        print("Creating features...")
        X = predictor.create_features(df)
        y = predictor.prepare_target(df)
        
        # Train model
        print("Training model...")
        metrics = predictor.train_model(X, y)
        print(f"Model R² score: {metrics['r2']:.4f}")
        
        # Analyze feature importance
        print("Analyzing feature importance...")
        feature_importance = predictor.analyze_feature_importance(X)
        
        # Analyze category patterns
        print("Analyzing category patterns...")
        category_patterns = predictor.analyze_category_patterns(df, X, y)
        
        # Save results
        results = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'category_patterns': category_patterns
        }
        predictor.save_results(results)
        
        # Create visualizations
        predictor.create_visualizations(results)
        
        print("Helpfulness analysis completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()