# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_in_the_blank_multilingual(text):
    """
    Take a text with [MASK] token(s) and predict the missing word in multiple languages.

    Args:
        text (str): A string containing the [MASK] token where the word is missing.

    Returns:
        list: A list of dictionaries with the predicted words and their scores.
    """
    # Initialize the multilingual masked language model
    unmasker = pipeline('fill-mask', model='distilbert-base-multilingual-cased')
    
    # Predict the word that fits the mask
    result = unmasker(text)
    return result


# test_function_code --------------------

def test_fill_in_the_blank_multilingual():
    print("Testing fill_in_the_blank_multilingual function.")

    # Testing with a multilingual sample sentence
    print("Testing case [1/1] started.")
    test_sentence = "This is a sample sentence with a [MASK] word."
    result = fill_in_the_blank_multilingual(test_sentence)
    assert len(result) > 0, 'No predictions returned.'
    assert isinstance(result, list), 'Result should be a list.'
    assert 'score' in result[0], 'Predicted result should contain a score.'
    assert 'sequence' in result[0], 'Predicted result should contain a sequence.'
    assert 'token_str' in result[0], 'Predicted result should contain a token string.'
    print("All test cases passed.")

    # Run the test
    test_fill_in_the_blank_multilingual()
