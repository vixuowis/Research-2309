# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_review_similarity(restaurant_reviews):
    """
    Calculate similarity scores for different restaurant reviews using SentenceTransformer.

    Args:
        restaurant_reviews (list of str): List of restaurant reviews.

    Returns:
        similarity_matrix (numpy.ndarray): A matrix of similarity scores.

    """
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    review_embeddings = model.encode(restaurant_reviews)
    similarity_matrix = cosine_similarity(review_embeddings)
    return similarity_matrix

# test_function_code --------------------

def test_calculate_review_similarity():
    """
    Test the calculate_review_similarity function.
    """
    reviews = ['The food was delicious', 'I loved the food', 'The food was not good']
    similarity_matrix = calculate_review_similarity(reviews)
    assert similarity_matrix.shape == (len(reviews), len(reviews))

# call_test_function_code --------------------

test_calculate_review_similarity()