# requirements_file --------------------

!pip install -U sentence-transformers scipy

# function_import --------------------

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# function_code --------------------

def calculate_similarity(sentence1, sentence2):
    """
    Calculate the similarity between two sentences using SentenceTransformer.

    Args:
    sentence1 (str): The first sentence.
    sentence2 (str): The second sentence.

    Returns:
    float: The similarity score between the two sentences, where a higher score indicates greater similarity.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    sentence1_embedding = model.encode(sentence1)
    sentence2_embedding = model.encode(sentence2)
    similarity = 1 - cosine(sentence1_embedding, sentence2_embedding)
    return similarity


# test_function_code --------------------

def test_calculate_similarity():
    print("Testing started.")

    # Test case 1: Identical sentences
    sentence1 = "This is a test sentence."
    sentence2 = "This is a test sentence."
    assert calculate_similarity(sentence1, sentence2) > 0.99, "Test case [1/3] failed: Identical sentences should have high similarity."

    # Test case 2: Completely different sentences
    sentence1 = "This is the first test sentence."
    sentence2 = "Different content altogether."
    assert calculate_similarity(sentence1, sentence2) < 0.2, "Test case [2/3] failed: Completely different sentences should have low similarity."

    # Test case 3: Partially similar sentences
    sentence1 = "The quick brown fox jumps over the lazy dog."
    sentence2 = "A fast dark-colored fox leaps above a slow dog."
    assert calculate_similarity(sentence1, sentence2) > 0.5, "Test case [3/3] failed: Partially similar sentences should have medium similarity."

    print("All tests passed.")

# Run the test function
test_calculate_similarity()
