# requirements_file --------------------

!pip install -U sentence-transformers sklearn

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_review_similarity(restaurant_reviews):
    """
    Calculate the similarity scores for different restaurant reviews using SentenceTransformer.

    Args:
        restaurant_reviews (list of str): A list of restaurant reviews to be compared.

    Returns:
        numpy.ndarray: A matrix with similarity scores between all pairs of reviews.

    Raises:
        ValueError: If restaurant_reviews is empty or not a list.
    """
    if not isinstance(restaurant_reviews, list) or not restaurant_reviews:
        raise ValueError("Input must be a non-empty list of reviews.")
    
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    review_embeddings = model.encode(restaurant_reviews)
    similarity_matrix = cosine_similarity(review_embeddings)
    return similarity_matrix

# test_function_code --------------------

def test_calculate_review_similarity():
    print("Testing started.")
    # Mock restaurant reviews
    restaurant_reviews = [
        'The food was fantastic and the service was superb!',
        'An amazing experience with delicious food.',
        'Average dining experience, nothing special.',
        'I did not like the food at all.',
        'Good service but the food was bland.'
    ]

    # Testing case 1: Valid input
    print("Testing case [1/2] started.")
    similarity_matrix = calculate_review_similarity(restaurant_reviews)
    assert isinstance(similarity_matrix, np.ndarray), f"Test case [1/2] failed: Expected similarity_matrix to be numpy.ndarray, got {type(similarity_matrix)}"
    
    # Testing case 2: Invalid input
    print("Testing case [2/2] started.")
    try:
        _ = calculate_review_similarity([])
        assert False, "Test case [2/2] failed: ValueError expected for empty input"
    except ValueError as e:
        assert str(e) == "Input must be a non-empty list of reviews.", f"Test case [2/2] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_calculate_review_similarity()