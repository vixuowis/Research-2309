# requirements_file --------------------

pip install -U sentence-transformers scipy

# function_import --------------------

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# function_code --------------------

def calculate_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the semantic similarity between two sentences.

    Args:
        sentence1: The first sentence to compare.
        sentence2: The second sentence to compare.

    Returns:
        A float representing the similarity score between the two sentences.

    Raises:
        ValueError: If any of the input sentences are empty.
    """
    # Validate the input sentences
    if not sentence1 or not sentence2:
        raise ValueError('Input sentences must not be empty.')

    # Load the pre-trained model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

    # Encode the sentences
    sentence1_embedding = model.encode(sentence1)
    sentence2_embedding = model.encode(sentence2)

    # Calculate cosine similarity
    similarity = 1 - cosine(sentence1_embedding, sentence2_embedding)

    return similarity

# test_function_code --------------------

def test_calculate_similarity():
    print('Testing started.')
    # Test case 1: Similar sentences
    print('Testing case [1/3] started.')
    sentence1 = 'The quick brown fox jumps over the lazy dog.'
    sentence2 = 'A fast dark-colored fox leaps above a sluggish canine.'
    similarity = calculate_similarity(sentence1, sentence2)
    assert similarity > 0.7, f'Test case [1/3] failed: Expected similarity above 0.7, got {similarity}'

    # Test case 2: Dissimilar sentences
    print('Testing case [2/3] started.')
    sentence1 = 'The quick brown fox jumps over the lazy dog.'
    sentence2 = 'I had a great time at the beach yesterday.'
    similarity = calculate_similarity(sentence1, sentence2)
    assert similarity < 0.4, f'Test case [2/3] failed: Expected similarity below 0.4, got {similarity}'

    # Test case 3: Empty sentence
    print('Testing case [3/3] started.')
    sentence1 = ''
    sentence2 = 'This is a non-empty sentence.'
    try:
        calculate_similarity(sentence1, sentence2)
        assert False, 'Test case [3/3] failed: Expected ValueError for empty sentence.'
    except ValueError:
        pass  # Expected
    print('Testing finished.')

# call_test_function_line --------------------

test_calculate_similarity()