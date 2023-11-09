# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def encode_sentences(sentences):
    """
    This function encodes a list of sentences into a 768-dimensional dense vector space using the SentenceTransformer model.
    
    Args:
        sentences (list): A list of sentences to be encoded.
    
    Returns:
        embeddings (list): A list of encoded sentences in a 768-dimensional dense vector space.
    """
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    embeddings = model.encode(sentences)
    return embeddings

# test_function_code --------------------

def test_encode_sentences():
    """
    This function tests the encode_sentences function by encoding a list of sentences and checking the output.
    """
    sentences = ['This is an example sentence.', 'Each sentence is converted.']
    embeddings = encode_sentences(sentences)
    assert len(embeddings) == len(sentences), 'Number of embeddings does not match number of sentences.'
    assert len(embeddings[0]) == 768, 'Embedding dimension does not match expected dimension.'

# call_test_function_code --------------------

test_encode_sentences()