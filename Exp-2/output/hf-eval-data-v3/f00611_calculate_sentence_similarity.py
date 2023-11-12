# function_import --------------------

from sentence_transformers import SentenceTransformer
import numpy as np

# function_code --------------------

def calculate_sentence_similarity(question1: str, question2: str) -> float:
    """
    Calculate the similarity between two sentences using SentenceTransformer.

    Args:
        question1 (str): The first sentence.
        question2 (str): The second sentence.

    Returns:
        float: The similarity score between the two sentences.
    """
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    embedding1 = model.encode(question1)
    embedding2 = model.encode(question2)
    similarity = np.inner(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

# test_function_code --------------------

def test_calculate_sentence_similarity():
    assert abs(calculate_sentence_similarity('What time is it?', 'Can you tell me the current time?') - 0.8) < 0.1
    assert abs(calculate_sentence_similarity('How are you?', 'What is your name?') - 0.2) < 0.1
    assert abs(calculate_sentence_similarity('Hello world', 'Hello world') - 1.0) < 0.1
    return 'All Tests Passed'

# call_test_function_code --------------------

test_calculate_sentence_similarity()