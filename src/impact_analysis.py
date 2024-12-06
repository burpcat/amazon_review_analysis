"""
Analyzes business impact and quantifies relationships between review characteristics,
sentiment, topics, and helpfulness with category-based topic organization.
"""

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from datetime import datetime
import os

class ImpactAnalyzer:
    def __init__(self, save_dir='impact_analysis'):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def load_all_results(self):
        """Load results from all previous analyses with category-based structure."""
        results = {}
        
        # Load processed reviews
        processed_files = glob.glob('processed_data/processed_reviews_*.csv')
        if processed_files:
            results['processed_reviews'] = pd.read_csv(max(processed_files))
        
        # Load category-specific topic terms
        results['topic_terms'] = {}
        category_dirs = glob.glob('topic_models/*/')
        for category_dir in category_dirs:
            category_name = os.path.basename(os.path.dirname(category_dir))
            terms_files = glob.glob(f'{category_dir}/top_terms_*.txt')
            if terms_files:
                with open(max(terms_files), 'r') as f:
                    # Skip the header line
                    next(f)
                    terms = f.read().strip().split(', ')
                    results['topic_terms'][category_name] = terms
        
        return results

    def analyze_category_topic_sentiment(self, df, topic_terms):
        """Analyze relationships between topics and sentiment for each category."""
        correlations = {}
        
        for category, terms in topic_terms.items():
            # Filter reviews for this category
            category_df = df[df['predicted_category'] == category]
            
            if len(category_df) == 0:
                continue
                
            # Calculate term presence in reviews
            term_presence = pd.DataFrame()
            for term in terms:
                term_presence[term] = category_df['clean_text'].str.contains(term, regex=False).astype(int)
            
            # Calculate correlations with sentiment
            term_correlations = []
            for term in terms:
                correlation = np.corrcoef(term_presence[term], 
                                        category_df['sentiment_score'])[0,1]
                term_correlations.append({'term': term, 'correlation': correlation})
            
            correlations[category] = pd.DataFrame(term_correlations)
        
        # Visualize correlations for each category
        for category, corr_df in correlations.items():
            plt.figure(figsize=(12, 6))
            sns.barplot(data=corr_df.sort_values('correlation'), 
                       x='correlation', y='term')
            plt.title(f'Term-Sentiment Correlations for {category}')
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/term_sentiment_correlations_{category.lower().replace(' & ', '_')}.png")
            plt.close()
        
        return correlations

    def calculate_business_impact(self, df):
        """Calculate business impact metrics with category focus."""
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
        
        # Add category-specific metrics
        impact_metrics['category_metrics'] = {}
        for category in df['predicted_category'].unique():
            category_df = df[df['predicted_category'] == category]
            impact_metrics['category_metrics'][category] = {
                'avg_sentiment': category_df['sentiment_score'].mean(),
                'avg_helpfulness': category_df['helpfulness_ratio'].mean(),
                'avg_rating': category_df['Score'].mean(),
                'review_count': len(category_df)
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
        """Measure impact of sentiment on ratings by category."""
        results = {}
        for category in df['predicted_category'].unique():
            category_df = df[df['predicted_category'] == category]
            optimal_mask = (category_df['sentiment_score'] >= 0.2) & (category_df['sentiment_score'] <= 0.4)
            
            if len(category_df[optimal_mask]) > 0:
                optimal_mean = category_df.loc[optimal_mask, 'Score'].mean()
                other_mean = category_df.loc[~optimal_mask, 'Score'].mean()
                pooled_std = np.sqrt((category_df.loc[optimal_mask, 'Score'].var() + 
                                    category_df.loc[~optimal_mask, 'Score'].var()) / 2)
                
                results[category] = {
                    'mean_improvement': optimal_mean - other_mean,
                    'effect_size': (optimal_mean - other_mean) / pooled_std if pooled_std != 0 else 0
                }
        
        return results

    def analyze_category_performance(self, df):
        """Analyze performance patterns by category."""
        return df.groupby('predicted_category').agg({
            'helpfulness_ratio': ['mean', 'std'],
            'sentiment_score': ['mean', 'std'],
            'Score': ['mean', 'count']
        }).round(3)

    def analyze_temporal_patterns(self, df):
        """Analyze temporal patterns by category."""
        df['Time'] = pd.to_datetime(df['Time'])
        return df.groupby(['predicted_category', df['Time'].dt.to_period('M')]).agg({
            'helpfulness_ratio': 'mean',
            'sentiment_score': 'mean',
            'Score': ['mean', 'count']
        }).round(3)

    def generate_recommendations(self, impact_metrics):
        """Generate category-specific recommendations."""
        recommendations = {}
        
        for category, metrics in impact_metrics['category_metrics'].items():
            category_recs = {
                'review_length': {
                    'target': f"{impact_metrics['review_characteristics']['optimal_length']['range'][0]}-"
                             f"{impact_metrics['review_characteristics']['optimal_length']['range'][1]} characters",
                    'expected_improvement': f"{impact_metrics['review_characteristics']['optimal_length']['helpfulness_increase']['mean_improvement']:.2%}"
                },
                'current_metrics': {
                    'avg_sentiment': f"{metrics['avg_sentiment']:.2f}",
                    'avg_helpfulness': f"{metrics['avg_helpfulness']:.2%}",
                    'avg_rating': f"{metrics['avg_rating']:.1f}",
                    'review_volume': metrics['review_count']
                }
            }
            
            # Add category-specific recommendations
            if metrics['avg_helpfulness'] < 0.5:
                category_recs['improvement_areas'] = ['helpfulness']
            if metrics['avg_sentiment'] < 0:
                category_recs['improvement_areas'] = ['sentiment']
            
            recommendations[category] = category_recs
        
        return recommendations

    def save_results(self, results_dict):
        """Save analysis results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, data in results_dict.items():
            if isinstance(data, dict):
                pd.DataFrame.from_dict(data, orient='index').to_csv(
                    f'{self.save_dir}/{name}_{timestamp}.csv'
                )
            elif isinstance(data, pd.DataFrame):
                data.to_csv(f'{self.save_dir}/{name}_{timestamp}.csv')

def main():
    """Run complete impact analysis."""
    try:
        print("Initializing impact analyzer...")
        analyzer = ImpactAnalyzer()
        
        print("Loading analysis results...")
        results = analyzer.load_all_results()
        
        if 'processed_reviews' not in results:
            raise FileNotFoundError("No processed reviews found!")
            
        df = results['processed_reviews']
        
        print("Analyzing category-topic-sentiment relationships...")
        topic_sentiment = analyzer.analyze_category_topic_sentiment(
            df, 
            results['topic_terms']
        )
        
        print("Calculating business impact...")
        impact_metrics = analyzer.calculate_business_impact(df)
        
        print("Generating recommendations...")
        recommendations = analyzer.generate_recommendations(impact_metrics)
        
        print("Saving results...")
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