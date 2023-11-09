def test_cluster_customer_reviews():
    # Test dataset
    reviews = ['This product is great!', 'I love this product!', 'This is the worst product I have ever bought.', 'I hate this product.', 'This product is okay, but it has some problems.']
    # Call the function with the test dataset
    labels = cluster_customer_reviews(reviews)
    # Check that the function returns a list
    assert isinstance(labels, np.ndarray), 'The function should return a numpy array.'
    # Check that the length of the returned list is equal to the length of the input list
    assert len(labels) == len(reviews), 'The length of the returned list should be equal to the length of the input list.'
    # Check that the returned list contains only integers (cluster labels)
    assert all(isinstance(label, np.int64) for label in labels), 'The returned list should contain only integers (cluster labels).'

test_cluster_customer_reviews()