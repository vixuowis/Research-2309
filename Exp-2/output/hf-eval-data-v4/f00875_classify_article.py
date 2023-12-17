# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_article(text, candidate_labels):
    """
    Classifies the political category of a given article.

    Args:
        text (str): Text of the article to classify.
        candidate_labels (list): A list of candidate labels for classification.

    Returns:
        dict: A dictionary containing the label and score of the predicted category.
    """
    # Create a zero-shot classification model instance
    zero_shot_classifier = pipeline('zero-shot-classification', model='MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary')
    # Classify the given text
    classification_result = zero_shot_classifier(text, candidate_labels, multi_label=False)
    return classification_result

# test_function_code --------------------

def test_classify_article():
    print("Testing classify_article function.")
    text = 'Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU.'
    candidate_labels = ['politics', 'economy', 'entertainment', 'environment']

    # Test classification
    result = classify_article(text, candidate_labels)
    assert result['labels'][0] == 'politics', f"Test failed: the correct category should be 'politics', got {result['labels'][0]} instead."
    print("Test passed: The article was correctly classified as politics.")

# Run test function
 test_classify_article()