"""
product_category_classifier.py

This module implements a product category classifier for Amazon reviews
using text-based classification approach.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import re

class ProductCategoryClassifier:
    def __init__(self):
        """Initialize the classifier with expanded category keywords."""
        # Define expanded category keywords mapping
        self.category_keywords = {
            'Food & Beverages': [
                # General Food Terms
                'food', 'snack', 'grocery', 'meal', 'gourmet', 'organic',
                # Beverages
                'drink', 'coffee', 'tea', 'juice', 'water', 'soda', 'beverage', 'wine', 'beer', 'alcohol',
                # Snacks
                'chocolate', 'candy', 'chips', 'crackers', 'nuts', 'popcorn', 'cookies',
                # Breakfast
                'cereal', 'oatmeal', 'breakfast', 'granola', 'pancake',
                # Condiments
                'sauce', 'spice', 'herb', 'seasoning', 'oil', 'vinegar', 'dressing',
                # Dairy & Alternatives
                'milk', 'cheese', 'yogurt', 'butter', 'cream', 'dairy'
            ],
            
            'Pet Supplies': [
                # General Pet Terms
                'pet', 'animal', 'veterinary', 'breed',
                # Dogs
                'dog', 'puppy', 'canine', 'leash', 'collar', 'harness', 'kennel',
                # Cats
                'cat', 'kitten', 'feline', 'litter', 'scratching', 'catnip',
                # Pet Food
                'food', 'treat', 'kibble', 'nutrition', 'feed', 'dietary',
                # Pet Accessories
                'toy', 'bowl', 'bed', 'cage', 'aquarium', 'tank',
                # Pet Care
                'grooming', 'shampoo', 'brush', 'nail', 'dental', 'medication'
            ],
            
            'Beauty & Personal Care': [
                # Skincare
                'skin', 'face', 'cream', 'lotion', 'moisturizer', 'serum', 'cleanser',
                # Hair Care
                'hair', 'shampoo', 'conditioner', 'styling', 'brush', 'dryer',
                # Makeup
                'makeup', 'cosmetic', 'lipstick', 'mascara', 'foundation', 'eyeshadow',
                # Personal Care
                'soap', 'deodorant', 'toothpaste', 'mouthwash', 'dental', 'hygiene',
                # Fragrance
                'perfume', 'cologne', 'fragrance', 'scent',
                # Tools
                'brush', 'applicator', 'mirror', 'spa', 'salon'
            ],
            
            'Health & Wellness': [
                # Supplements
                'vitamin', 'supplement', 'mineral', 'protein', 'omega', 'antioxidant',
                # Fitness
                'fitness', 'exercise', 'workout', 'gym', 'weight', 'muscle',
                # Medical
                'medicine', 'medical', 'health', 'pill', 'tablet', 'capsule',
                # Natural Health
                'organic', 'natural', 'herbal', 'essential oil', 'homeopathic',
                # Medical Devices
                'monitor', 'thermometer', 'pressure', 'brace', 'support',
                # Wellness
                'wellness', 'meditation', 'yoga', 'relaxation', 'sleep'
            ],
            
            'Home & Kitchen': [
                # Kitchen
                'kitchen', 'cookware', 'bakeware', 'utensil', 'appliance', 'pot', 'pan',
                # Dining
                'dish', 'plate', 'bowl', 'cup', 'glass', 'cutlery', 'dinnerware',
                # Cleaning
                'clean', 'vacuum', 'mop', 'broom', 'detergent', 'soap',
                # Storage
                'storage', 'container', 'organizer', 'basket', 'box', 'shelf',
                # Furniture
                'furniture', 'chair', 'table', 'bed', 'sofa', 'desk',
                # Home Decor
                'decor', 'curtain', 'rug', 'lamp', 'pillow', 'blanket'
            ],
            
            'Electronics': [
                # Computers
                'computer', 'laptop', 'desktop', 'monitor', 'keyboard', 'mouse',
                # Mobile Devices
                'phone', 'smartphone', 'tablet', 'ipad', 'android', 'ios',
                # Audio
                'headphone', 'speaker', 'earbud', 'audio', 'sound', 'microphone',
                # Video
                'tv', 'television', 'screen', 'projector', 'camera', 'video',
                # Gaming
                'game', 'gaming', 'console', 'controller', 'playstation', 'xbox',
                # Accessories
                'charger', 'cable', 'adapter', 'battery', 'case', 'stand'
            ],
            
            'Sports & Outdoors': [
                # Sports Equipment
                'sport', 'ball', 'bat', 'racket', 'golf', 'basketball',
                # Outdoor Recreation
                'camping', 'hiking', 'tent', 'backpack', 'outdoor', 'fishing',
                # Exercise Equipment
                'exercise', 'gym', 'weight', 'treadmill', 'yoga', 'fitness',
                # Clothing
                'athletic', 'sportswear', 'shoe', 'apparel', 'gear',
                # Water Sports
                'swim', 'pool', 'beach', 'surf', 'kayak', 'boat',
                # Winter Sports
                'ski', 'snowboard', 'winter', 'ice', 'snow'
            ],
            
            'Office & School Supplies': [
                # Writing Supplies
                'pen', 'pencil', 'marker', 'highlighter', 'eraser',
                # Paper Products
                'paper', 'notebook', 'folder', 'binder', 'calendar',
                # Office Equipment
                'printer', 'scanner', 'copier', 'shredder', 'laminator',
                # Desk Accessories
                'stapler', 'tape', 'clip', 'organizer', 'holder',
                # School Supplies
                'school', 'backpack', 'calculator', 'student', 'education',
                # Art Supplies
                'art', 'craft', 'paint', 'brush', 'canvas'
            ],
            
            'Tools & Home Improvement': [
                # Hand Tools
                'tool', 'hammer', 'screwdriver', 'wrench', 'plier',
                # Power Tools
                'drill', 'saw', 'sander', 'grinder', 'power tool',
                # Hardware
                'nail', 'screw', 'bolt', 'nut', 'washer',
                # Paint & Supplies
                'paint', 'brush', 'roller', 'tape', 'primer',
                # Electrical
                'electrical', 'wire', 'outlet', 'switch', 'light',
                # Plumbing
                'plumbing', 'pipe', 'faucet', 'sink', 'toilet'
            ],
            
            'Clothing & Accessories': [
                # Men's Clothing
                'mens', 'man', 'shirt', 'pants', 'jacket', 'suit',
                # Women's Clothing
                'womens', 'woman', 'dress', 'skirt', 'blouse', 'top',
                # Children's Clothing
                'kids', 'child', 'baby', 'boy', 'girl', 'infant',
                # Shoes
                'shoe', 'boot', 'sneaker', 'sandal', 'heel', 'footwear',
                # Accessories
                'watch', 'jewelry', 'belt', 'wallet', 'bag', 'purse',
                # Seasonal
                'swimwear', 'winter', 'summer', 'spring', 'fall'
            ]
        }
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        
        self.negative_keywords = {
                    'Pet Supplies': [
                        'human food', 'snack food', 'candy', 'chocolate', 'beverage',
                        'drink', 'coffee', 'tea', 'juice', 'soda', 'cooking',
                        'baking', 'ingredient', 'spice', 'herb', 'oil', 'sauce'
            ]
        }
    
    def train(self):
        """Train the classifier using synthetic data."""
        print("Creating synthetic training data...")
        texts, labels = self._create_training_data()
        
        # Transform texts to TF-IDF features
        print("Converting texts to TF-IDF features...")
        X = self.vectorizer.fit_transform(texts)
        
        # Initialize and train the classifier
        print("Training the classifier...")
        self.classifier = MultinomialNB()
        self.classifier.fit(X, labels)
        
        print("Training complete!")
            
    def _create_training_data(self):
        """Create synthetic training data using keyword mappings."""
        training_texts = []
        training_labels = []
        
        # Generate synthetic review texts for each category
        for category, keywords in self.category_keywords.items():
            # Create multiple combinations of keywords for each category
            for _ in range(200):  # Increased number of synthetic samples
                # Randomly select 3-5 keywords
                num_keywords = np.random.randint(3, 6)
                selected_keywords = np.random.choice(keywords, num_keywords, replace=False)
                
                # Create synthetic reviews with more varied structures
                synthetic_text = f"This is a {category.lower()} review about {' and '.join(selected_keywords)}. "
                synthetic_text += f"The {np.random.choice(selected_keywords)} quality is good. "
                
                # Add category-specific phrases
                if category == 'Pet Supplies':
                    synthetic_text += f"My pet loves this {np.random.choice(selected_keywords)}. "
                
                training_texts.append(synthetic_text)
                training_labels.append(category)
                
                # Add negative examples using negative keywords
                if category in self.negative_keywords:
                    for _ in range(50):  # Add some negative examples
                        negative_text = f"This {np.random.choice(self.negative_keywords[category])} "
                        negative_text += "is not related to pets. "
                        training_texts.append(negative_text)
                        training_labels.append('Other')
        
        return training_texts, training_labels
    
    def categorize_reviews(self, df):
        """Categorize reviews with confidence threshold."""
        print("Starting review categorization...")
        
        # Combine Summary and Text for better classification
        combined_texts = df['Summary'] + ' ' + df['Text']
        
        # Transform texts
        print("Converting reviews to TF-IDF features...")
        X = self.vectorizer.transform(combined_texts)
        
        # Predict categories and probabilities
        print("Predicting categories...")
        categories = self.classifier.predict(X)
        probabilities = np.max(self.classifier.predict_proba(X), axis=1)
        
        # Apply confidence threshold
        df['predicted_category'] = categories
        df['category_confidence'] = probabilities
        
        # Mark low-confidence predictions as 'Uncategorized'
        low_confidence_mask = df['category_confidence'] < self.confidence_threshold
        df.loc[low_confidence_mask, 'predicted_category'] = 'Uncategorized'
        
        # Print category distribution
        print("\nCategory Distribution after confidence threshold:")
        print(df['predicted_category'].value_counts())
        print(f"\nTotal uncategorized due to low confidence: {low_confidence_mask.sum()}")
        
        return df

def main():
    """Main function to demonstrate usage."""
    # Load the processed reviews
    print("Loading processed reviews...")
    df = pd.read_csv('processed_data/processed_reviews_latest.csv')
    
    # Initialize and train classifier
    classifier = ProductCategoryClassifier()
    classifier.train()
    
    # Categorize all reviews
    df = classifier.categorize_reviews(df)
    
    # Save categorized reviews
    output_file = 'processed_data/categorized_reviews.csv'
    df.to_csv(output_file, index=False)
    print(f"Categorized reviews saved to {output_file}")
    
    # Print category distribution
    print("\nCategory Distribution:")
    print(df['predicted_category'].value_counts())

if __name__ == "__main__":
    main()