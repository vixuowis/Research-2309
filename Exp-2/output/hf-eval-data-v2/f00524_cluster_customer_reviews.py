# function_import --------------------

from sentence_transformers import SentenceTransformer

# function_code --------------------

def cluster_customer_reviews(reviews):
    """
    This function clusters customer reviews based on their content similarity.
    
    Args:
        reviews (list): A list of customer reviews.
    
    Returns:
        embeddings (list): A list of embeddings for each review.
    
    Raises:
        ValueError: If the input is not a list or if it's empty.
    """
    if not isinstance(reviews, list) or not reviews:
        raise ValueError('Input should be a non-empty list of reviews.')
    
    model = SentenceTransformer('nikcheerla/nooks-amd-detection-v2-full')
    embeddings = model.encode(reviews)
    return embeddings

# test_function_code --------------------

def test_cluster_customer_reviews():
    """
    This function tests the cluster_customer_reviews function.
    """
    test_reviews = ['This is a great product.', 'I am not satisfied with the product.', 'The product is okay.']
    embeddings = cluster_customer_reviews(test_reviews)
    assert isinstance(embeddings, list), 'The result should be a list.'
    assert len(embeddings) == len(test_reviews), 'The number of embeddings should be equal to the number of reviews.'

# call_test_function_code --------------------

test_cluster_customer_reviews()