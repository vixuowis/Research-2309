from transformers import pipeline

def classify_news_article(news_article: str) -> str:
    """
    Classify a news article into categories like 'Politics', 'Sports', 'Technology', 'Business', and 'Entertainment'
    using the Hugging Face Transformers library.

    Args:
        news_article (str): The text of the news article to classify.

    Returns:
        str: The category that the news article belongs to.
    """
    classifier = pipeline('zero-shot-classification', model='typeform/squeezebert-mnli')
    candidate_labels = ['Politics', 'Sports', 'Technology', 'Business', 'Entertainment']
    result = classifier(news_article, candidate_labels)
    return result['labels'][0]