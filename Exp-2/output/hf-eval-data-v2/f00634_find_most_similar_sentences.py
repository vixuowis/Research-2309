# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# function_code --------------------

def find_most_similar_sentences(sentences):
    """
    Analyze a set of sentences to find the most similar pairs using SentenceTransformer.

    Args:
        sentences (list): A list of sentences to analyze.

    Returns:
        tuple: A tuple containing two most similar sentences and their similarity score.
    """
    model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')
    embeddings = model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, 0)
    indices = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
    return (sentences[indices[0]], sentences[indices[1]], similarity_matrix[indices])

# test_function_code --------------------

def test_find_most_similar_sentences():
    """
    Test the function find_most_similar_sentences.
    """
    sentences = ['I have a dog', 'My dog loves to play', 'There is a cat in our house', 'The cat and the dog get along well']
    result = find_most_similar_sentences(sentences)
    assert len(result) == 3, 'The result should be a tuple of three elements.'
    assert isinstance(result[0], str) and isinstance(result[1], str), 'The first two elements of the result should be strings.'
    assert isinstance(result[2], float), 'The third element of the result should be a float.'

# call_test_function_code --------------------

test_find_most_similar_sentences()