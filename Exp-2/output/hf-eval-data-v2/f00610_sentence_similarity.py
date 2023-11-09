# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def sentence_similarity(sentences):
    """
    This function takes a list of sentences and returns a similarity matrix.
    The similarity is calculated using the SentenceTransformer model from Hugging Face Transformers.

    Args:
        sentences (list): A list of sentences for which to calculate similarity.

    Returns:
        similarity_matrix (numpy.ndarray): A matrix where the element at the i-th row and j-th column represents the similarity between the i-th and j-th sentences.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    embeddings = model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

# test_function_code --------------------

def test_sentence_similarity():
    """
    This function tests the sentence_similarity function.
    It uses a small set of sentences and checks if the similarity matrix is symmetric and the diagonal elements are 1.
    """
    sentences = ['This is an example sentence.', 'Each sentence is converted.', 'This is another similar sentence.']
    similarity_matrix = sentence_similarity(sentences)
    assert similarity_matrix.shape == (len(sentences), len(sentences)), 'The shape of the similarity matrix is incorrect.'
    assert (similarity_matrix.diagonal() == 1).all(), 'The diagonal elements of the similarity matrix should be 1.'
    assert (similarity_matrix == similarity_matrix.T).all(), 'The similarity matrix should be symmetric.'

# call_test_function_code --------------------

test_sentence_similarity()