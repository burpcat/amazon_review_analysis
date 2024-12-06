"""
impact_analysis.py

Analyzes business impact and quantifies relationships between review characteristics,
sentiment, topics, and helpfulness.
"""

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from datetime import datetime
import os
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

class ImpactAnalyzer:
    def __init__(self, save_dir='impact_analysis'):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def load_all_results(self):
        """Load results from all previous analyses."""
        results = {}
        
        # Load topic distributions
        topic_files = glob.glob('topic_models/topic_distributions_*.csv')
        if topic_files:
            results['topic_distributions'] = pd.read_csv(max(topic_files))
            
        # Load sentiment results
        sentiment_files = glob.glob('advanced_sentiment/temporal_trends_*.csv')
        if sentiment_files:
            results['sentiment_trends'] = pd.read_csv(max(sentiment_files))
            
        # Load helpfulness results
        helpfulness_files = glob.glob('helpfulness_analysis/feature_importance_*.csv')
        if helpfulness_files:
            results['feature_importance'] = pd.read_csv(max(helpfulness_files))
            
        return results

    def analyze_topic_sentiment_relationships(self, topic_distributions, sentiment_scores):
        """Analyze relationships between topics and sentiment."""
        correlations = pd.DataFrame(index=['correlation'])
        for topic in topic_distributions.columns:
            correlations[topic] = np.corrcoef(topic_distributions[topic], 
                                            sentiment_scores)[0,1]
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        sns.heatmap(correlations, annot=True, cmap='RdYlBu', center=0)
        plt.title('Topic-Sentiment Correlations')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/topic_sentiment_correlations.png")
        return correlations

    def calculate_business_impact(self, df):
        """Calculate business impact metrics."""
        impact_metrics = {
            'review_characteristics': {
                'optimal_length': {
                    'range': (2000, 5000),
                    'helpfulness_increase': self.measure_length_impact(df)
                },
                'optimal_sentiment': {
                    'range': (0.2, 0.4),
                    'rating_increase': self.measure_sentiment_impact(df)
                }
            },
            'category_performance': self.analyze_category_performance(df),
            'temporal_patterns': self.analyze_temporal_patterns(df)
        }
        return impact_metrics

    def measure_length_impact(self, df):
        """Measure impact of review length on helpfulness."""
        optimal_mask = (df['text_length'] >= 2000) & (df['text_length'] <= 5000)
        optimal_mean = df.loc[optimal_mask, 'helpfulness_ratio'].mean()
        other_mean = df.loc[~optimal_mask, 'helpfulness_ratio'].mean()
        pooled_std = np.sqrt((df.loc[optimal_mask, 'helpfulness_ratio'].var() + 
                            df.loc[~optimal_mask, 'helpfulness_ratio'].var()) / 2)
    
        return {
            'mean_improvement': optimal_mean - other_mean,
            'effect_size': (optimal_mean - other_mean) / pooled_std if pooled_std != 0 else 0
        }

    def measure_sentiment_impact(self, df):
        """Measure impact of sentiment on ratings."""
        optimal_mask = (df['sentiment_score'] >= 0.2) & (df['sentiment_score'] <= 0.4)
        optimal_mean = df.loc[optimal_mask, 'Score'].mean()
        other_mean = df.loc[~optimal_mask, 'Score'].mean()
        pooled_std = np.sqrt((df.loc[optimal_mask, 'Score'].var() + 
                            df.loc[~optimal_mask, 'Score'].var()) / 2)
    
        return {
            'mean_improvement': optimal_mean - other_mean,
            'effect_size': (optimal_mean - other_mean) / pooled_std if pooled_std != 0 else 0
        }
    
    def analyze_category_performance(self, df):
        """Analyze performance patterns by category."""
        return df.groupby('ProductId').agg({
            'helpfulness_ratio': ['mean', 'std'],
            'sentiment_score': ['mean', 'std'],
            'Score': ['mean', 'count']
        }).round(3)

    def analyze_temporal_patterns(self, df):
        """Analyze temporal patterns in review impact."""
        df['Time'] = pd.to_datetime(df['Time'])
        return df.groupby(df['Time'].dt.to_period('M')).agg({
            'helpfulness_ratio': ['mean', 'std'],
            'sentiment_score': ['mean', 'std'],
            'Score': ['mean', 'count']
        }).round(3)

    def generate_recommendations(self, impact_metrics):
        """Generate actionable business recommendations."""
        recommendations = {
            'review_length': {
                'target': f"{impact_metrics['review_characteristics']['optimal_length']['range'][0]}-"
                         f"{impact_metrics['review_characteristics']['optimal_length']['range'][1]} characters",
                'expected_improvement': f"{impact_metrics['review_characteristics']['optimal_length']['helpfulness_increase']['mean_improvement']:.2%}"
            },
            'sentiment_balance': {
                'target': f"{impact_metrics['review_characteristics']['optimal_sentiment']['range'][0]}-"
                         f"{impact_metrics['review_characteristics']['optimal_sentiment']['range'][1]} sentiment score",
                'expected_improvement': f"{impact_metrics['review_characteristics']['optimal_sentiment']['rating_increase']['mean_improvement']:.2%}"
            },
            'category_focus': self.identify_priority_categories(impact_metrics['category_performance'])
        }
        return recommendations

    def identify_priority_categories(self, category_performance):
        """Identify categories needing attention."""
        return {
            'high_potential': category_performance[
                (category_performance[('Score', 'mean')] > 4.0) &
                (category_performance[('helpfulness_ratio', 'mean')] < 0.5)
            ].index.tolist(),
            'needs_improvement': category_performance[
                (category_performance[('Score', 'mean')] < 3.5) &
                (category_performance[('helpfulness_ratio', 'mean')] < 0.3)
            ].index.tolist()
        }

    def save_results(self, results_dict):
        """Save analysis results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, data in results_dict.items():
            if isinstance(data, pd.DataFrame):
                data.to_csv(f'{self.save_dir}/{name}_{timestamp}.csv')
            else:
                pd.DataFrame([data]).to_csv(f'{self.save_dir}/{name}_{timestamp}.csv')

def main():
    """Run complete impact analysis."""
    try:
        analyzer = ImpactAnalyzer()
        
        # Load processed data with correct filename
        print("Loading analysis results...")
        latest_processed = max(glob.glob('processed_data/processed_reviews_*.csv'))
        df = pd.read_csv(latest_processed)
        results = analyzer.load_all_results()
        
        # Continue with remaining analysis...
        print("Analyzing topic-sentiment relationships...")
        topic_sentiment = analyzer.analyze_topic_sentiment_relationships(
            results['topic_distributions'],
            df['sentiment_score']
        )
        
        # Calculate business impact
        print("Calculating business impact...")
        impact_metrics = analyzer.calculate_business_impact(df)
        
        # Generate recommendations
        print("Generating recommendations...")
        recommendations = analyzer.generate_recommendations(impact_metrics)
        
        # Save results
        results_dict = {
            'topic_sentiment_correlations': topic_sentiment,
            'impact_metrics': impact_metrics,
            'recommendations': recommendations
        }
        analyzer.save_results(results_dict)
        
        print("Impact analysis completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()