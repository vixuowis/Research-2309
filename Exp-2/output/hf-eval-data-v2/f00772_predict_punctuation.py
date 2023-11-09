# function_import --------------------

from transformers import pipeline

# function_code --------------------

def predict_punctuation(novel_draft_text):
    """
    Predicts the punctuation marks needed in a given text.

    Args:
        novel_draft_text (str): The text in which to predict punctuation marks.

    Returns:
        dict: A dictionary where keys are the words in the text and values are the predicted punctuation marks.
    """
    punctuation_predictor = pipeline('token-classification', model='kredor/punctuate-all')
    predicted_punctuations = punctuation_predictor(novel_draft_text)
    return predicted_punctuations

# test_function_code --------------------

def test_predict_punctuation():
    """
    Tests the predict_punctuation function.
    """
    test_text = 'Hello world This is a test text'
    expected_output = {'Hello': '', 'world': '.', 'This': '', 'is': '', 'a': '', 'test': '', 'text': '.'}
    assert predict_punctuation(test_text) == expected_output

# call_test_function_code --------------------

test_predict_punctuation()