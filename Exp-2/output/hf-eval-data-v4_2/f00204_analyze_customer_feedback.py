# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_customer_feedback(feedback_text):
    “”“Analyze the sentiment of the given feedback text using pre-trained sentiment analysis model.

    Args:
        feedback_text (str): The text of the customer feedback in Spanish.

    Returns:
        dict: A dictionary containing the sentiment classification result.

    Raises:
        ValueError: If the feedback_text is not a string or is empty.
    “”“
    if not isinstance(feedback_text, str) or not feedback_text:
        raise ValueError('The feedback text must be a non-empty string.')
    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
    sentiment = sentiment_task(feedback_text)
    return sentiment[0]

# test_function_code --------------------

def test_analyze_customer_feedback():
    print("Testing started.")
    
    # Test case 1: Valid feedback text
    print("Testing case [1/3] started.")
    sentiment_result = analyze_customer_feedback("Me encanta este producto!")
    assert sentiment_result['label'] in ['LABEL_0', 'LABEL_1', 'LABEL_2'], f"Test case [1/3] failed: Expected a valid sentiment label, got {sentiment_result['label']} instead."
    
    # Test case 2: Empty feedback text
    print("Testing case [2/3] started.")
    try:
        analyze_customer_feedback("")
        assert False, "Test case [2/3] failed: ValueError expected for empty input."
    except ValueError as e:
        assert str(e) == 'The feedback text must be a non-empty string.', f"Test case [2/3] failed: Incorrect error message: {str(e)}"
    
    # Test case 3: Non-string feedback text
    print("Testing case [3/3] started.")
    try:
        analyze_customer_feedback(None)
        assert False, "Test case [3/3] failed: ValueError expected for non-string input."
    except ValueError as e:
        assert str(e) == 'The feedback text must be a non-empty string.', f"Test case [3/3] failed: Incorrect error message: {str(e)}"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_customer_feedback()