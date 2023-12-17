# requirements_file --------------------

import subprocess

requirements = ["sentence-transformers", "sklearn", "numpy"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the similarity between two sentences using a pre-trained transformer model.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The similarity score between the two sentences.

    Raises:
        ValueError: If any of the sentences is empty.
    """
    if not sentence1 or not sentence2:
        raise ValueError("One of the sentences is empty.")

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    embeddings = model.encode([sentence1, sentence2])
    similarity_score = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))
    return similarity_score[0][0]

# test_function_code --------------------

def test_calculate_sentence_similarity():
    print("Testing started.")

    # Test case 1: Two identical sentences
    print("Testing case [1/3] started.")
    assert calculate_sentence_similarity("Hello world", "Hello world") == 1, f"Test case [1/3] failed: The similarity score should be 1 for identical sentences."

    # Test case 2: Two different sentences
    print("Testing case [2/3] started.")
    score = calculate_sentence_similarity("Good morning", "Good night")
    assert 0 <= score <= 1, f"Test case [2/3] failed: The similarity score should be between 0 and 1."

    # Test case 3: Testing with empty string
    print("Testing case [3/3] started.")
    try:
        calculate_sentence_similarity("", "This is a test.")
        assert False, f"Test case [3/3] failed: An empty string should raise a ValueError."
    except ValueError as e:
        assert str(e) == "One of the sentences is empty.", f"Test case [3/3] failed: A ValueError should be raised with the message 'One of the sentences is empty.'"

    print("Testing finished.")

# call_test_function_line --------------------

test_calculate_sentence_similarity()