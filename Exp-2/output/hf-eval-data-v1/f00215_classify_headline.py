from transformers import pipeline

def classify_headline(headline: str):
    """
    This function classifies a given news headline into one of three categories: sports, technology, or politics.
    It uses the zero-shot classification model 'cross-encoder/nli-deberta-v3-xsmall' from the transformers library.
    
    Parameters:
    headline (str): The news headline to classify.
    
    Returns:
    str: The predicted category of the news headline.
    """
    # Create a classifier using the pipeline function from the transformers library
    headlines_classifier = pipeline('zero-shot-classification', model='cross-encoder/nli-deberta-v3-xsmall')
    
    # Define the candidate labels
    candidate_labels = ['technology', 'sports', 'politics']
    
    # Use the classifier to predict the category of the news headline
    headline_category = headlines_classifier(headline, candidate_labels)
    
    # Return the predicted category
    return headline_category['labels'][0]