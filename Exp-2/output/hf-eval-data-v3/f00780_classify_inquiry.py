# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_inquiry(inquiry: str) -> str:
    """
    Classify a customer inquiry into one of the following categories: "sales", "technical support", or "billing".

    Args:
        inquiry (str): The customer inquiry to be classified.

    Returns:
        str: The category of the inquiry, which can be either "sales", "technical support", or "billing".
    """
    classifier = pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli')
    candidate_labels = ['sales', 'technical support', 'billing']
    hypothesis_template = 'The inquiry is related to {}.'
    result = classifier(inquiry, candidate_labels, hypothesis_template=hypothesis_template)
    return result['labels'][0]

# test_function_code --------------------

def test_classify_inquiry():
    """
    Test the classify_inquiry function.
    """
    test_inquiry1 = 'I am experiencing difficulty with the installation process of your software.'
    expected_result1 = 'technical support'
    assert classify_inquiry(test_inquiry1) == expected_result1

    test_inquiry2 = 'I want to buy your product, can you give me a discount?'
    expected_result2 = 'sales'
    assert classify_inquiry(test_inquiry2) == expected_result2

    test_inquiry3 = 'I have a problem with my bill, it seems too high.'
    expected_result3 = 'billing'
    assert classify_inquiry(test_inquiry3) == expected_result3

    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_inquiry()