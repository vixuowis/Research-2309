# function_import --------------------

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the similarity between two sentences using the SentenceTransformer model.

    Args:
        sentence1 (str): The first sentence to compare.
        sentence2 (str): The second sentence to compare.

    Returns:
        float: The similarity score between the two sentences.
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    embeddings = model.encode([sentence1, sentence2])
    similarity_score = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))
    return similarity_score[0][0]

# test_function_code --------------------

def test_calculate_sentence_similarity():
    assert abs(calculate_sentence_similarity('This is the first sentence.', 'This is the second sentence.') - 0.8) < 0.1
    assert abs(calculate_sentence_similarity('I love apples.', 'I love oranges.') - 0.9) < 0.1
    assert abs(calculate_sentence_similarity('The sky is blue.', 'The grass is green.') - 0.7) < 0.1
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_calculate_sentence_similarity())