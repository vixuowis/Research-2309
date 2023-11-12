# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def generate_embeddings(sentences):
    '''
    Generate embeddings for the input sentences using SentenceTransformer model.

    Args:
        sentences (list): A list of sentences for which to generate embeddings.

    Returns:
        numpy.ndarray: A 2D array where each row represents the embedding of a sentence.
    '''
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    embeddings = model.encode(sentences)
    return embeddings

# test_function_code --------------------

def test_generate_embeddings():
    '''
    Test the generate_embeddings function.
    '''
    sentences = ['This is an example sentence', 'Each sentence is converted']
    embeddings = generate_embeddings(sentences)
    assert embeddings.shape[0] == len(sentences), 'Number of embeddings should be equal to number of sentences'
    assert embeddings.shape[1] == 768, 'Each embedding should have 768 dimensions'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_embeddings()