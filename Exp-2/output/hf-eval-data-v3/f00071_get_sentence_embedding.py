# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def get_sentence_embedding(sentence: str) -> list:
    '''
    This function takes a sentence as input and returns its embedding using the SentenceTransformer model.

    Args:
        sentence (str): The sentence to be encoded.

    Returns:
        list: The embedding of the input sentence.
    '''
    model = SentenceTransformer('sentence-transformers/nli-mpnet-base-v2')
    encoded_sentence = model.encode(sentence)
    return encoded_sentence

# test_function_code --------------------

def test_get_sentence_embedding():
    '''
    This function tests the get_sentence_embedding function.
    '''
    sentence1 = 'The effects of climate change on biodiversity and ecosystem services in the Arctic.'
    sentence2 = 'Climate change is a significant threat to biodiversity in the Arctic.'
    assert len(get_sentence_embedding(sentence1)) == 768
    assert len(get_sentence_embedding(sentence2)) == 768
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_sentence_embedding()