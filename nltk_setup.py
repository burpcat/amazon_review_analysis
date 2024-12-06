"""
nltk_setup.py - Run this script once to download required NLTK resources
"""

import nltk

def download_nltk_resources():
    """Download all required NLTK resources."""
    print("Downloading NLTK resources...")
    resources = [
        'punkt',
        'averaged_perceptron_tagger',
        'stopwords',
        'wordnet'
    ]
    
    for resource in resources:
        print(f"Downloading {resource}...")
        nltk.download(resource)
    
    print("All NLTK resources downloaded successfully!")

if __name__ == "__main__":
    download_nltk_resources()