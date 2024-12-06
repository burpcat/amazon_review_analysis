# Amazon Review Analysis Project

## Overview
This project implements a comprehensive analysis system for Amazon product reviews, including sentiment analysis, topic modeling, helpfulness prediction, and business impact analysis.

## Features
- **Data Processing**: Clean and preprocess Amazon review data
- **Sentiment Analysis**: Advanced sentiment analysis using DistilBERT
- **Topic Modeling**: Category-specific topic modeling with coherence optimization
- **Helpfulness Prediction**: Machine learning model to predict review helpfulness
- **Impact Analysis**: Business impact analysis and recommendations
- **Visualizations**: Comprehensive data visualizations

## Directory Structure
```
project/
├── data/
│   └── Reviews.csv
├── src/
│   ├── __init__.py
│   ├── amazon_review_processor.py
│   ├── helpfulness_predictor.py
│   ├── impact_analysis.py
│   ├── nltk_setup.py
│   ├── product_category_classifier.py
│   ├── sentiment_analysis.py
│   ├── topic_modeling.py
│   └── visualization.py
├── processed_data/
├── topic_models/
├── advanced_sentiment/
├── helpfulness_analysis/
├── impact_analysis/
├── visualizations/
├── driver.py
├── README.md
└── .gitignore
```

## Prerequisites
- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - torch
  - transformers
  - nltk
  - textblob
  - lightgbm
  - seaborn
  - matplotlib
  - plotly
  - gensim
  - tqdm

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd amazon-review-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv .usmlenv
source .usmlenv/bin/activate  # On Windows: .usmlenv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Download NLTK resources:
```bash
python src/nltk_setup.py
```

## Usage
1. Place your Amazon review dataset (`Reviews.csv`) in the `data/` directory

2. Run the complete analysis pipeline:
```bash
python driver.py
```

## Module Descriptions
- **amazon_review_processor.py**: Data loading, cleaning, and initial processing
- **sentiment_analysis.py**: Deep learning-based sentiment analysis
- **topic_modeling.py**: Category-specific topic modeling
- **helpfulness_predictor.py**: ML model for helpfulness prediction
- **impact_analysis.py**: Business impact analysis
- **visualization.py**: Data visualization tools
- **product_category_classifier.py**: Product categorization
- **nltk_setup.py**: NLTK resource downloader

## Output
The analysis pipeline generates several outputs:
- Processed review data
- Topic modeling results by category
- Sentiment analysis results
- Helpfulness predictions
- Business impact metrics
- Various visualizations

## License
This project is licensed under the MIT License - see the LICENSE file for details.