# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_in_the_blanks(sentence: str) -> str:
    '''
    Fill in the blanks in a sentence using a pre-trained BERT model.

    Args:
        sentence (str): The sentence with a '[MASK]' token representing the missing word.

    Returns:
        str: The sentence with the '[MASK]' token replaced by the predicted word.
    '''
    fill_mask = pipeline('fill-mask', model='bert-large-uncased')
    filled_sentence = fill_mask(sentence)
    return filled_sentence

# test_function_code --------------------

def test_fill_in_the_blanks():
    '''
    Test the fill_in_the_blanks function.
    '''
    assert fill_in_the_blanks('The cat chased the [MASK] around the house.') == 'The cat chased the dog around the house.'
    assert fill_in_the_blanks('I want to [MASK] a book.') == 'I want to read a book.'
    assert fill_in_the_blanks('He is a [MASK] student.') == 'He is a good student.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_fill_in_the_blanks()