# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_fill_in_the_blank_questions(masked_sentence):
    """
    This function generates fill-in-the-blank questions using a pre-trained model.

    Args:
        masked_sentence (str): The sentence with a keyword replaced by the '[MASK]' token.

    Returns:
        list: A list of possible words that fit the masked position.
    """
    unmasker = pipeline('fill-mask', model='distilbert-base-multilingual-cased')
    possible_words = unmasker(masked_sentence)
    return possible_words

# test_function_code --------------------

def test_generate_fill_in_the_blank_questions():
    """
    This function tests the 'generate_fill_in_the_blank_questions' function.
    It uses a sample sentence and checks if the returned list of possible words is not empty.
    """
    test_sentence = "Hello, I'm a [MASK] model."
    result = generate_fill_in_the_blank_questions(test_sentence)
    assert len(result) > 0, "The list of possible words is empty."
    print('Test passed.')

# call_test_function_code --------------------

test_generate_fill_in_the_blank_questions()