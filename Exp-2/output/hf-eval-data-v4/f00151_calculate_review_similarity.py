# requirements_file --------------------

!pip install -U sentence-transformers sklearn

# function_import --------------------

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# function_code --------------------

def calculate_review_similarity(reviews):
    """
    Calculate similarity scores between pairs of restaurant reviews.

    Args:
        reviews (list of str): A list of restaurant reviews as strings.

    Returns:
        numpy.ndarray: A matrix of similarity scores.
    """
    # Load the model
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')

    # Convert reviews to embeddings
    review_embeddings = model.encode(reviews)

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(review_embeddings)

    return similarity_matrix

# test_function_code --------------------

def test_calculate_review_similarity():
    print("Testing calculate_review_similarity function.")
    reviews = [
        'The food was great!',
        'Absolutely loved the food and the service!',
        'Not great, but the food was decent.',
        'Would not recommend. The food was cold and service was slow.'
    ]

    # Expected shape of similarity matrix
    expected_shape = (len(reviews), len(reviews))

    # Calculate similarity scores
    similarity_scores = calculate_review_similarity(reviews)

    # Test for the shape of the similarity matrix
    assert similarity_scores.shape == expected_shape, f"Test failed: Similarity matrix shape expected to be {expected_shape} but was {similarity_scores.shape}."

    print("All tests passed!")

# Running the test
    test_calculate_review_similarity()