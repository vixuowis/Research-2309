# function_import --------------------

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the similarity between two sentences using SentenceTransformer.

    Args:
        sentence1 (str): The first sentence to compare.
        sentence2 (str): The second sentence to compare.

    Returns:
        float: The similarity score between the two sentences. The score is between -1 (completely dissimilar) and 1 (completely similar).
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    embeddings = model.encode([sentence1, sentence2])
    similarity_score = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))
    return similarity_score[0][0]

# test_function_code --------------------

def test_calculate_sentence_similarity():
    """
    Test the calculate_sentence_similarity function.
    """
    sentence1 = 'This is the first sentence.'
    sentence2 = 'This is the second sentence.'
    sentence3 = 'This is an entirely different sentence.'
    assert 0.8 <= calculate_sentence_similarity(sentence1, sentence2) <= 1.0
    assert -1.0 <= calculate_sentence_similarity(sentence1, sentence3) <= 0.2

# call_test_function_code --------------------

test_calculate_sentence_similarity()