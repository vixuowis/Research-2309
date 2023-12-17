# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import XLMRobertaForSequenceClassification

# function_code --------------------

def categorize_inquiry(inquiry: str) -> str:
    """
    Classify a customer inquiry into one of the predefined categories.

    Args:
        inquiry: The text of the customer inquiry.

    Returns:
        The category the inquiry belongs to.

    """
    classifier = XLMRobertaForSequenceClassification.from_pretrained('joeddav/xlm-roberta-large-xnli')
    candidate_labels = ['sales', 'technical support', 'billing']
    hypothesis_template = 'This inquiry is related to {}.'

    # Call the classifier providing the inquiry, candidate labels, and the hypothesis_template
    result = classifier(inquiry, candidate_labels, hypothesis_template=hypothesis_template)

    # Obtain the most likely category
    category = max(result['labels'], key=lambda label: result['scores'][result['labels'].index(label)])

    return category

# test_function_code --------------------

def test_categorize_inquiry():
    print("Testing started.")

    # Test case: Correct classification of a sales-related inquiry
    print("Testing case [1/3] started.")
    assert categorize_inquiry("I want to purchase your product.") == "sales", "Test case [1/3] failed: should be classified as sales."

    # Test case: Correct classification of a technical support-related inquiry
    print("Testing case [2/3] started.")
    assert categorize_inquiry("My device won't start.") == "technical support", "Test case [2/3] failed: should be classified as technical support."

    # Test case: Correct classification of a billing-related inquiry
    print("Testing case [3/3] started.")
    assert categorize_inquiry("I have a question about my invoice.") == "billing", "Test case [3/3] failed: should be classified as billing."
    print("Testing finished.")

# call_test_function_line --------------------

test_categorize_inquiry()