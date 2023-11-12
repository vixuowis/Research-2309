# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def encode_sentences(sentences):
    """
    This function encodes a list of sentences into a 768-dimensional dense vector space using SentenceTransformer.

    Args:
        sentences (list): A list of sentences to be encoded.

    Returns:
        numpy.ndarray: An array of encoded sentences.
    """
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    embeddings = model.encode(sentences)
    return embeddings

# test_function_code --------------------

def test_encode_sentences():
    """
    This function tests the `encode_sentences` function with some test cases.
    """
    test_sentences = ['This is a test sentence.', 'Another test sentence.']
    embeddings = encode_sentences(test_sentences)
    assert embeddings.shape == (2, 768), 'Test case 1 failed'
    test_sentences = ['One more test sentence.']
    embeddings = encode_sentences(test_sentences)
    assert embeddings.shape == (1, 768), 'Test case 2 failed'
    print('All tests passed')

# call_test_function_code --------------------

test_encode_sentences()