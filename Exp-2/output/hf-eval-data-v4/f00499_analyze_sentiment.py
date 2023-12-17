# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text using zero-shot classification.

    Parameters:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary with 'label' and 'score' indicating the sentiment.
    """
    nlp = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-6')
    result = nlp(text, ['positive', 'negative'])
    return result

# test_function_code --------------------

def test_analyze_sentiment():
    print("Testing analyze_sentiment function.")
    # Example positive text
    positive_text = "The new technology has a groundbreaking potential to change the world for the better."
    positive_result = analyze_sentiment(positive_text)
    assert positive_result['labels'][0] == 'positive', f"Expected positive sentiment, but got {positive_result['labels'][0]}"

    # Example negative text
    negative_text = "This technology is flawed and fails to deliver on its promises."
    negative_result = analyze_sentiment(negative_text)
    assert negative_result['labels'][0] == 'negative', f"Expected negative sentiment, but got {negative_result['labels'][0]}"

    print("All tests passed!")

# Run the test function
test_analyze_sentiment()