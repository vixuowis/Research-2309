# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def compute_sentence_embeddings(sentences):
    """
    Compute the embeddings for a set of sentences using the SentenceTransformer.

    Args:
        sentences (list): A list of sentences for which the embeddings are to be computed.

    Returns:
        numpy.ndarray: An array of embeddings for the input sentences.
    """
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # or any pretrained sentence transformer
    return model.encode(list(sentences))

# test_function_code --------------------

def test_compute_sentence_embeddings():
    """
    Test the compute_sentence_embeddings function.
    """
    sentences = ["This is an example sentence", "Each sentence is converted"]
    embeddings = compute_sentence_embeddings(sentences)
    assert embeddings.shape[0] == len(sentences), 'The number of embeddings should be equal to the number of sentences.'
    assert embeddings.shape[1] == 768, 'The dimension of each embedding should be 768.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_compute_sentence_embeddings()