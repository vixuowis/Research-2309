from transformers import pipeline

def classify_inquiry(inquiry):
    """
    Classify a customer inquiry into one of the following categories: "sales", "technical support", or "billing".

    Args:
        inquiry (str): The customer inquiry to be classified.

    Returns:
        str: The category of the inquiry.
    """
    classifier = pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli')
    candidate_labels = ['sales', 'technical support', 'billing']
    result = classifier(inquiry, candidate_labels)
    return result['labels'][0]