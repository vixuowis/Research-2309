# function_import --------------------

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# function_code --------------------

def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the similarity between two sentences using SentenceTransformer model.

    Args:
        sentence1 (str): The first sentence to compare.
        sentence2 (str): The second sentence to compare.

    Returns:
        float: The similarity score between the two sentences. A high score indicates that the two sentences are likely to be semantically similar.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    sentence1_embedding = model.encode(sentence1)
    sentence2_embedding = model.encode(sentence2)
    similarity = 1 - cosine(sentence1_embedding, sentence2_embedding)
    return similarity

# test_function_code --------------------

def test_calculate_sentence_similarity():
    """
    Test the function calculate_sentence_similarity.
    """
    sentence1 = 'This is an example sentence'
    sentence2 = 'Each sentence is converted'
    similarity = calculate_sentence_similarity(sentence1, sentence2)
    assert 0 <= similarity <= 1, 'The similarity score should be between 0 and 1.'

# call_test_function_code --------------------

test_calculate_sentence_similarity()