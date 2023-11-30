# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_text(text_message: str, candidate_labels: list) -> dict:
    """
    Classify a given text message into one of the provided categories.

    Args:
        text_message (str): The text message to be classified.
        candidate_labels (list): A list of potential categories for the text message.

    Returns:
        dict: A dictionary containing the classification results.

    Raises:
        OSError: If there is a problem with the model loading due to disk quota exceeded.
    """

    try:
        classifier = pipeline("zero-shot-classification")
        result = classifier(text_message, candidate_labels)
        
        return {"predictions": result}
    
    except OSError as e:
        raise e

# test_function_code --------------------

def test_classify_text():
    """
    Test the classify_text function with some example text messages and candidate labels.
    """
    text_message1 = 'Your monthly bank statement is now available.'
    candidate_labels1 = ['finances', 'health', 'entertainment']
    assert isinstance(classify_text(text_message1, candidate_labels1), dict)

    text_message2 = 'Remember to take your vitamins.'
    candidate_labels2 = ['health', 'finances', 'entertainment']
    assert isinstance(classify_text(text_message2, candidate_labels2), dict)

    text_message3 = 'The new movie is out in theaters.'
    candidate_labels3 = ['entertainment', 'finances', 'health']
    assert isinstance(classify_text(text_message3, candidate_labels3), dict)

    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_text()