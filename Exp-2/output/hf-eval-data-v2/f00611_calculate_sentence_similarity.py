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
        float: The cosine similarity between the two sentence embeddings.
    """
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    embedding1 = model.encode(question1)
    embedding2 = model.encode(question2)
    similarity = np.inner(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

# test_function_code --------------------

def test_calculate_sentence_similarity():
    """
    Test the function calculate_sentence_similarity.
    """
    question1 = 'What time is it?'
    question2 = 'Can you tell me the current time?'
    question3 = 'What is your name?'
    similarity1 = calculate_sentence_similarity(question1, question2)
    similarity2 = calculate_sentence_similarity(question1, question3)
    assert 0.7 <= similarity1 <= 1.0, 'Expected similarity score between 0.7 and 1.0'
    assert 0.0 <= similarity2 <= 0.3, 'Expected similarity score between 0.0 and 0.3'

# call_test_function_code --------------------

test_calculate_sentence_similarity()