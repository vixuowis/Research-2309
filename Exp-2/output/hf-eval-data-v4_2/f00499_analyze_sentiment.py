# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(text, model='valhalla/distilbart-mnli-12-6'):
    """
    Determine if the sentiment of a given text is positive or negative using zero-shot classification.

    Args:
        text (str): The text to be analyzed.
        model (str): The model to be used for analysis. Default is 'valhalla/distilbart-mnli-12-6'.

    Returns:
        dict: A dictionary with the analysis results, including labels and scores.

    Raises:
        ValueError: If the input text is empty.
    """
    if not text:
        raise ValueError('Input text cannot be empty.')
    classifier = pipeline('zero-shot-classification', model=model)
    return classifier(text, ['positive', 'negative'])

# test_function_code --------------------

def test_analyze_sentiment():
    print("Testing started.")
    # Test case 1: positive text
    print("Testing case [1/2] started.")
    positive_result = analyze_sentiment('The new technology is groundbreaking and innovative.')
    assert positive_result['labels'][0] == 'positive', f"Test case [1/2] failed: {positive_result}"
    
    # Test case 2: negative text
    print("Testing case [2/2] started.")
    negative_result = analyze_sentiment('This technology is obsolete and disappointing.')
    assert negative_result['labels'][0] == 'negative', f"Test case [2/2] failed: {negative_result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_sentiment()