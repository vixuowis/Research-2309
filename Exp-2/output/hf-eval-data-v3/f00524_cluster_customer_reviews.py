# function_import --------------------

import numpy as np
from sentence_transformers import SentenceTransformer

# function_code --------------------

def cluster_customer_reviews(reviews):
    """
    Cluster customer reviews based on their content similarity.

    Args:
        reviews (list): A list of customer reviews.

    Returns:
        embeddings (list): A list of embeddings for each review.

    Raises:
        ValueError: If reviews is not a list or is empty.
    """
    if not isinstance(reviews, list) or not reviews:
        raise ValueError('Input reviews should be a non-empty list.')

    model = SentenceTransformer('nikcheerla/nooks-amd-detection-v2-full')
    embeddings = model.encode(reviews)
    return embeddings

# test_function_code --------------------

def test_cluster_customer_reviews():
    """
    Test the function cluster_customer_reviews.
    """
    reviews = ['This is a great product.', 'I am not satisfied with the product.', 'The product is affordable and of high quality.']
    embeddings = cluster_customer_reviews(reviews)
    assert isinstance(embeddings, list), 'The result should be a list.'
    assert len(embeddings) == len(reviews), 'The length of embeddings should be equal to the length of reviews.'
    assert all(isinstance(e, np.ndarray) for e in embeddings), 'Each element in embeddings should be a numpy array.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_cluster_customer_reviews()