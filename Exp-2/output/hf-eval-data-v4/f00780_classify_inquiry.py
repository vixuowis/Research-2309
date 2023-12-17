# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import XLMRobertaForSequenceClassification
from transformers import pipeline

# function_code --------------------

def classify_inquiry(inquiry_text, candidate_labels):
    # Load pre-trained XLM-Roberta large model for zero-shot classification
    classifier = pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli')

    # Perform classification
    results = classifier(inquiry_text, candidate_labels)

    # Extract the most probable category
    category = results['labels'][0]
    return category

# test_function_code --------------------

def test_classify_inquiry():
    print("Testing classify_inquiry function.")
    candidate_labels = ["sales", "technical support", "billing"]
    # Test case: Technical support inquiry
    inquiry = "I am experiencing difficulty with the installation process of your software."
    expected_category = "technical support"
    assert classify_inquiry(inquiry, candidate_labels) == expected_category, "Test case failed: Expected technical support category."
    print("Test case passed: Technical support inquiry correctly classified.")

    # More test cases can be added as needed.

    print("Testing completed.")

# Run the test function
test_classify_inquiry()