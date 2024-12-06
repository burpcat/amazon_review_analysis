from src import sentiment_analysis, amazon_review_processor, helpfulness_predictor, impact_analysis, nltk_setup, topic_modeling, visualization

if __name__ == "__main__":
    nltk_setup.download_nltk_resources()
    amazon_review_processor.review_processor_runner()
    topic_modeling.main()
    sentiment_analysis.main()
    helpfulness_predictor.main()
    impact_analysis.main()
    visualization.main()
