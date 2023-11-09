# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_language(text):
    """
    Detects the language of the given text using a pre-trained model.

    Args:
        text (str): The text whose language is to be detected.

    Returns:
        dict: A dictionary containing the detected language and its confidence score.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')

    language_detection = pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')
    result = language_detection(text)
    return result

# test_function_code --------------------

def test_detect_language():
    """
    Tests the detect_language function with some test cases.

    Raises:
        AssertionError: If the test fails.
    """
    test_text_english = 'Hello, how are you?'
    test_text_french = 'Bonjour, comment ça va?'
    result_english = detect_language(test_text_english)
    result_french = detect_language(test_text_french)

    assert result_english[0]['label'] == 'en', 'Test case for English text failed.'
    assert result_french[0]['label'] == 'fr', 'Test case for French text failed.'

# call_test_function_code --------------------

test_detect_language()