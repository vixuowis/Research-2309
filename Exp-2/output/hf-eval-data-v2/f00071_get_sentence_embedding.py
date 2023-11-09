# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def get_sentence_embedding(sentence: str) -> list:
    """
    This function takes a sentence as input and returns its embedding.

    Args:
        sentence (str): The sentence to be encoded.

    Returns:
        list: The embedding of the input sentence.
    """
    model = SentenceTransformer('sentence-transformers/nli-mpnet-base-v2')
    encoded_sentence = model.encode(sentence)
    return encoded_sentence

# test_function_code --------------------

def test_get_sentence_embedding():
    """
    This function tests the 'get_sentence_embedding' function.
    It uses a sample sentence and checks if the output is a list.
    """
    sample_sentence = 'The effects of climate change on biodiversity and ecosystem services in the Arctic.'
    result = get_sentence_embedding(sample_sentence)
    assert isinstance(result, list), 'The result should be a list.'

# call_test_function_code --------------------

test_get_sentence_embedding()