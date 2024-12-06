"""
This module implements advanced sentiment analysis techniques for Amazon product reviews,
including deep learning models, temporal analysis, and category-specific insights.
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import os
from datetime import datetime
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):  # Kept original max_length
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

class AdvancedSentimentAnalyzer:
    def __init__(self, model_name='distilbert-base-uncased', sample_size=50000):  # Kept original sample_size
        """Initialize the advanced sentiment analyzer."""
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model with CPU optimizations
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            torch_dtype=torch.float32
        ).to(self.device)
        
        # Enable torch inference mode for better CPU performance
        torch.inference_mode(True)
        
        self.sample_size = sample_size
        
    def load_processed_data(self):
        """Load the most recent processed data."""
        processed_files = glob.glob('processed_data/processed_reviews_*.csv')
        
        if not processed_files:
            raise FileNotFoundError("Processed data files not found.")
            
        latest_file = max(processed_files)
        
        # Read only necessary columns and use a more efficient data type
        df = pd.read_csv(
            latest_file, 
            usecols=['Text', 'Score', 'Time', 'ProductId'],
            dtype={'Score': 'int8', 'Text': 'string'}
        )
        print(f"Loaded processed data from: {latest_file}")
        
        # Sample the data
        if len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
            print(f"Sampled {self.sample_size} reviews from dataset")
        
        return df

    def prepare_data(self, df):
        """Prepare data for deep learning model."""
        df['sentiment_label'] = pd.cut(
            df['Score'],
            bins=[-np.inf, 2, 3, np.inf],
            labels=[0, 1, 2]
        )
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df['Text'].values,
            df['sentiment_label'].values,
            test_size=0.2,
            random_state=42
        )
        
        print(f"Training samples: {len(train_texts)}, Test samples: {len(test_texts)}")
        return train_texts, test_texts, train_labels, test_labels

    def train_model(self, train_texts, train_labels, batch_size=16, epochs=2):  # Smaller batch size for CPU
        """Train the deep learning model with CPU optimizations."""
        # Create dataset and dataloader
        train_dataset = ReviewDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Single worker for CPU
            pin_memory=False  # No pin_memory for CPU
        )
        
        # Training settings
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=2e-5,
            eps=1e-8
        )
        
        # Use gradient accumulation to simulate larger batch size
        accumulation_steps = 8  # Effective batch size = 16 * 8 = 128
        
        # Training loop with progress bar
        self.model.train()
        for epoch in range(epochs):
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            total_loss = 0
            optimizer.zero_grad()
            
            for i, batch in enumerate(progress_bar):
                # Move batch to CPU (explicit casting to save memory)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Calculate loss and backpropagate
                loss = outputs.loss / accumulation_steps
                loss.backward()
                
                # Update weights after accumulation
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps
                
                # Update progress bar and free memory
                progress_bar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
                del outputs
                torch.cuda.empty_cache()  # Won't affect CPU but keeps code consistent
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def analyze_temporal_trends(self, df):
        """Analyze sentiment trends over time."""
        print("Analyzing temporal trends...")
        df['Time'] = pd.to_datetime(df['Time'])
    
        # Group by month and calculate statistics
        monthly_trends = (df.groupby(df['Time'].dt.to_period('M'))
                        .agg({
                            'sentiment_score': ['mean', 'std'],
                            'Score': ['mean', 'count']
                        })
                        .reset_index())
    
        return monthly_trends

    def analyze_category_patterns(self, df):
        """Analyze sentiment patterns by product category."""
        print("Analyzing category patterns...")
        # Group by product and calculate statistics
        patterns = (df.groupby('ProductId')
                .agg({
                    'sentiment_score': ['mean', 'std'],
                    'Score': ['mean', 'count']
                })
                .reset_index())
    
        # Filter to include only categories with sufficient data
        return patterns[patterns[('Score', 'count')] >= 10]

    def save_results(self, results_dict, directory='advanced_sentiment'):
        """Save analysis results."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        torch.save(self.model.state_dict(), 
                  f'{directory}/sentiment_model_{timestamp}.pth')
        
        # Save results
        for name, data in results_dict.items():
            if isinstance(data, pd.DataFrame):
                data.to_csv(f'{directory}/{name}_{timestamp}.csv')
        
        print(f"Results saved in {directory} directory with timestamp {timestamp}")

    def create_visualizations(self, results_dict, directory='advanced_sentiment'):
        """Create visualizations of the analysis results."""
        print("Creating visualizations...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
        if 'temporal_trends' in results_dict:
            plt.figure(figsize=(12, 6))
            trends_df = results_dict['temporal_trends']
            # Convert Period to datetime for plotting
            trends_df['Time'] = trends_df['Time'].astype(str).apply(pd.to_datetime)
            # Plot mean sentiment score over time
            sns.lineplot(
                data=trends_df,
                x='Time',
                y=('sentiment_score', 'mean'),
                label='Mean Sentiment'
            )
            plt.title('Sentiment Trends Over Time')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{directory}/temporal_trends_{timestamp}.png')
            plt.close()
        
        if 'category_patterns' in results_dict:
            plt.figure(figsize=(12, 6))
            patterns_df = results_dict['category_patterns']
            # Create boxplot for sentiment distribution
            sns.boxplot(
                data=patterns_df,
                x=('Score', 'mean'),
                y=('sentiment_score', 'mean')
            )
            plt.title('Sentiment Distribution by Average Rating')
            plt.tight_layout()
            plt.savefig(f'{directory}/category_patterns_{timestamp}.png')
            plt.close()

def main():
    """Main function to run advanced sentiment analysis."""
    try:
        print("Starting advanced sentiment analysis...")
        
        # Initialize analyzer with original sample size
        analyzer = AdvancedSentimentAnalyzer(sample_size=50000)
        
        # Load and prepare data
        df = analyzer.load_processed_data()
        train_texts, test_texts, train_labels, test_labels = analyzer.prepare_data(df)
        
        # Train model
        print("\nTraining sentiment model...")
        analyzer.train_model(train_texts, train_labels)
        
        # Add sentiment predictions to DataFrame
        print("\nGenerating sentiment scores...")
        dataset = ReviewDataset(df['Text'].values, np.zeros(len(df)), analyzer.tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_sentiments = []
        analyzer.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating sentiment scores"):
                input_ids = batch['input_ids'].to(analyzer.device)
                attention_mask = batch['attention_mask'].to(analyzer.device)
                outputs = analyzer.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                # Convert logits to sentiment scores (-1 to 1 range)
                sentiments = torch.softmax(logits, dim=1)
                # Use positive class probability as sentiment score
                sentiment_scores = sentiments[:, 2].cpu().numpy() - sentiments[:, 0].cpu().numpy()
                all_sentiments.extend(sentiment_scores)
        
        df['sentiment_score'] = all_sentiments
        
        # Now analyze patterns with the added sentiment scores
        print("\nAnalyzing temporal trends...")
        temporal_trends = analyzer.analyze_temporal_trends(df)
        category_patterns = analyzer.analyze_category_patterns(df)
        
        # Save results and create visualizations
        results = {
            'temporal_trends': temporal_trends,
            'category_patterns': category_patterns
        }
        analyzer.save_results(results)
        analyzer.create_visualizations(results)
        
        print("\nAdvanced sentiment analysis completed successfully!")
        
        return df  # Return the DataFrame with sentiment scores
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()