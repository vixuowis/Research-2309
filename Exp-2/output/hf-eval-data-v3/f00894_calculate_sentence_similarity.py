# function_import --------------------

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# function_code --------------------

def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the similarity between two sentences using SentenceTransformer model.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The similarity score between the two sentences. A high score indicates that the two sentences are likely to be semantically similar.

    Raises:
        ValueError: If the input sentences are not strings.
    """
    if not isinstance(sentence1, str) or not isinstance(sentence2, str):
        raise ValueError('Both inputs should be strings.')

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    sentence1_embedding = model.encode(sentence1)
    sentence2_embedding = model.encode(sentence2)
    similarity = 1 - cosine(sentence1_embedding, sentence2_embedding)
    return similarity

# test_function_code --------------------

def test_calculate_sentence_similarity():
    assert abs(calculate_sentence_similarity('This is a test.', 'This is a test.') - 1.0) < 0.01
    assert abs(calculate_sentence_similarity('This is a test.', 'This is not a test.') - 0.5) < 0.01
    assert abs(calculate_sentence_similarity('This is a test.', 'Completely different sentence.') - 0.0) < 0.01
    return 'All Tests Passed'

# call_test_function_code --------------------

test_calculate_sentence_similarity()