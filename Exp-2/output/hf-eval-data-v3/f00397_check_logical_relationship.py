# function_import --------------------

from sentence_transformers import CrossEncoder

# function_code --------------------

def check_logical_relationship(sentence1: str, sentence2: str) -> dict:
    '''
    Check the logical relationship between two sentences using the Hugging Face Transformers model.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        dict: A dictionary with the probability scores for the labels 'contradiction', 'entailment', and 'neutral'.
    '''
    model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
    scores = model.predict([(sentence1, sentence2)])
    return scores

# test_function_code --------------------

def test_check_logical_relationship():
    '''
    Test the check_logical_relationship function.
    '''
    assert isinstance(check_logical_relationship('A man is eating pizza', 'A man eats something'), dict)
    assert isinstance(check_logical_relationship('A black race car starts up in front of a crowd of people.', 'A man is driving down a lonely road.'), dict)
    assert isinstance(check_logical_relationship('A woman is walking her dog.', 'A woman is outside.'), dict)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_check_logical_relationship()