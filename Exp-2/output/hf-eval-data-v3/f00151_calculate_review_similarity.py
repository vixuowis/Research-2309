# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_review_similarity(reviews):
    """
    Calculate similarity scores for different restaurant reviews.

    Args:
        reviews (list of str): List of restaurant reviews.

    Returns:
        numpy.ndarray: Similarity matrix of the reviews.
    """
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    review_embeddings = model.encode(reviews)
    similarity_matrix = cosine_similarity(review_embeddings)
    return similarity_matrix

# test_function_code --------------------

def test_calculate_review_similarity():
    reviews = [
        'The food was delicious and the service was excellent.',
        'I loved the food and the staff was very friendly.',
        'The food was terrible and the service was poor.',
        'I did not enjoy the food and the staff was rude.']
    similarity_matrix = calculate_review_similarity(reviews)
    assert similarity_matrix.shape == (4, 4), 'The shape of the similarity matrix is not correct.'
    assert similarity_matrix[0, 1] > similarity_matrix[0, 2], 'The similarity score between the first two reviews should be higher than the score between the first and third reviews.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_calculate_review_similarity()