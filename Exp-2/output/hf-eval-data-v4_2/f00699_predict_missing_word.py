# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def predict_missing_word(text: str) -> str:
    """
    Predict the missing word in a multilingual text using a masked language model.

    Args:
        text (str): The input text with a '[MASK]' token where the word is missing.

    Returns:
        str: The predicted word that fits the context of the missing word.

    Raises:
        ValueError: If the input text does not contain the '[MASK]' token.
    """
    if '[MASK]' not in text:
        raise ValueError('The input text must contain a [MASK] token.')

    unmasker = pipeline('fill-mask', model='distilbert-base-multilingual-cased')
    result = unmasker(text)
    # Assuming the top result returned is the desired prediction
    predicted_word = result[0]['token_str']
    return predicted_word

# test_function_code --------------------

def test_predict_missing_word():
    print("Testing started.")

    # Test case 1: English language
    print("Testing case [1/3] started.")
    english_text = "Hello, I\'m a [MASK] model."
    predicted_word = predict_missing_word(english_text)
    assert predicted_word == 'language', f"Test case [1/3] failed: Expected 'language', got '{predicted_word}'"

    # Test case 2: French language
    print("Testing case [2/3] started.")
    french_text = "Bonjour, je suis un mod\u00e8le [MASK]."
    predicted_word = predict_missing_word(french_text)
    assert predicted_word == 'linguistique', f"Test case [2/3] failed: Expected 'linguistique', got '{predicted_word}'"

    # Test case 3: Test without MASK token
    print("Testing case [3/3] started.")
    invalid_text = "This text has no mask."
    try:
        predicted_word = predict_missing_word(invalid_text)
    except ValueError as e:
        assert str(e) == 'The input text must contain a [MASK] token.', f"Test case [3/3] failed: Expected a ValueError with specific message."

    print("Testing finished.")


# call_test_function_line --------------------

test_predict_missing_word()