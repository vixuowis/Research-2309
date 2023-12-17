# requirements_file --------------------

import subprocess

requirements = ["sentence-transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def calculate_sentence_similarity(sentence_one: str, sentence_two: str) -> float:
    """
    Calculate the cosine similarity between two sentences using a pre-trained SentenceTransformer model.

    Args:
        sentence_one (str): The first sentence for comparison.
        sentence_two (str): The second sentence for comparison.

    Returns:
        float: The cosine similarity between sentence_one and sentence_two.

    Raises:
        ValueError: If any of the sentences is empty.
    """
    if not sentence_one or not sentence_two:
        raise ValueError('Both sentences must be provided.')

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    embeddings = model.encode([sentence_one, sentence_two])
    cosine_similarity = embeddings[0].dot(embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))

    return cosine_similarity

# test_function_code --------------------

import numpy as np
def test_calculate_sentence_similarity():
    print("Testing started.")

    # Test case 1: Identical sentences
    print("Testing case [1/3] started.")
    identical_similarity = calculate_sentence_similarity("This is a test sentence.", "This is a test sentence.")
    assert np.isclose(identical_similarity, 1.0), f"Test case [1/3] failed: Expected similarity 1.0, but got {identical_similarity}"

    # Test case 2: Completely different sentences
    print("Testing case [2/3] started.")
    different_similarity = calculate_sentence_similarity("This is a test sentence.", "Different sentence entirely.")
    assert different_similarity < 1.0, f"Test case [2/3] failed: Expected similarity less than 1.0, but got {different_similarity}"

    # Test case 3: Empty sentence
    print("Testing case [3/3] started.")
    try:
        _ = calculate_sentence_similarity("", "This is a test sentence.")
        assert False, "Test case [3/3] failed: Expected ValueError for empty sentence"
    except ValueError as e:
        assert str(e) == 'Both sentences must be provided.', f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_calculate_sentence_similarity()