# function_import --------------------

from transformers import pipeline

# function_code --------------------

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

# test_function_code --------------------

def test_classify_inquiry():
    """
    Test the classify_inquiry function.
    """
    test_inquiry = 'I am experiencing difficulty with the installation process of your software.'
    expected_result = 'technical support'
    assert classify_inquiry(test_inquiry) == expected_result

# call_test_function_code --------------------

test_classify_inquiry()