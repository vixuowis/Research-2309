from transformers import pipeline


def news_category_detector(text):
    """
    This function detects whether a piece of news is talking about technology, sports, or politics.
    It uses the zero-shot-classification model from the transformers library.
    
    Args:
    text (str): The news text to be classified.
    
    Returns:
    str: The category of the news.
    """
    # Define the candidate labels
    candidate_labels = ["technology", "sports", "politics"]
    
    # Create a classifier pipeline with zero-shot-classification
    classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-roberta-base")
    
    # Classify the text
    result = classifier(text, candidate_labels)
    
    # Return the label with the highest score
    return result['labels'][result['scores'].index(max(result['scores']))]