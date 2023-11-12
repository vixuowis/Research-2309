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
        OSError: If there is a problem with the model loading due to disk quota exceeded.
    """
    try:
        language_detection = pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')
        result = language_detection(text)
        return result
    except OSError as e:
        print(f'Error: {e}')

# test_function_code --------------------

def test_detect_language():
    """
    Tests the detect_language function with different test cases.

    Returns:
        str: 'All Tests Passed' if all the assertions pass, else the function will raise an AssertionError.
    """
    test_text_english = 'Hello, how are you?'
    test_text_french = 'Bonjour, comment ça va?'
    test_text_german = 'Hallo, wie geht es dir?'

    result_english = detect_language(test_text_english)
    result_french = detect_language(test_text_french)
    result_german = detect_language(test_text_german)

    assert result_english[0]['label'] == 'LABEL_0', 'Test Case 1 Failed'
    assert result_french[0]['label'] == 'LABEL_1', 'Test Case 2 Failed'
    assert result_german[0]['label'] == 'LABEL_2', 'Test Case 3 Failed'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_language()