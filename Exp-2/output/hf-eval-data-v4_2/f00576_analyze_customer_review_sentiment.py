# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_review_sentiment(review_text):
    """
    Analyze the sentiment of a customer review text using a pre-trained model.

    Args:
        review_text (str): The text of the customer review to be analyzed.

    Returns:
        dict: A dictionary containing the label (POS, NEG, NEU) and score of the analyzed sentiment.

    Raises:
        ValueError: If review_text is not provided.
    """
    if not review_text:
        raise ValueError('The review_text argument must be provided')
    sentiment_model = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    sentiment_result = sentiment_model(review_text)
    return sentiment_result

# test_function_code --------------------

def test_analyze_customer_review_sentiment():
    print("Testing started.")
    # Test case 1: Positive sentiment
    print("Testing case [1/3] started.")
    result = analyze_customer_review_sentiment('Me encanta este producto, es increíble!')
    assert result[0]['label'] in ['POS'], f"Test case [1/3] failed: {result}"

    # Test case 2: Negative sentiment
    print("Testing case [2/3] started.")
    result = analyze_customer_review_sentiment('No me gusta este producto, es terrible.')
    assert result[0]['label'] in ['NEG'], f"Test case [2/3] failed: {result}"

    # Test case 3: Neutral sentiment
    print("Testing case [3/3] started.")
    result = analyze_customer_review_sentiment('Es un producto normal, nada especial.')
    assert result[0]['label'] in ['NEU'], f"Test case [3/3] failed: {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_customer_review_sentiment()