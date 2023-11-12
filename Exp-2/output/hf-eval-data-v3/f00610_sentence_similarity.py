# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def sentence_similarity(sentences):
    """
    This function takes a list of sentences and returns a matrix of cosine similarity scores.

    Args:
        sentences (list): A list of sentences.

    Returns:
        numpy.ndarray: A matrix of cosine similarity scores.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    embeddings = model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

# test_function_code --------------------

def test_sentence_similarity():
    sentences = ['This is an example sentence.', 'Each sentence is converted.', 'This is another similar sentence.']
    similarity_matrix = sentence_similarity(sentences)
    assert similarity_matrix.shape == (3, 3), 'The shape of the similarity matrix is not correct.'
    assert similarity_matrix[0, 0] == 1.0, 'The similarity score of a sentence with itself should be 1.0.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_sentence_similarity()