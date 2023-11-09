def test_classify_movie_reviews():
    # Test the classify_movie_reviews function with a sample dataset
    predictions = classify_movie_reviews('sample_data.csv')
    
    # Assert that the function returns a non-empty list
    assert len(predictions) > 0, 'The function should return a list of predictions.'
    
    # Assert that the function returns a list of strings (Positive or Negative)
    assert all(isinstance(prediction, str) for prediction in predictions), 'The function should return a list of strings.'
    
    # Assert that the function returns only 'Positive' or 'Negative'
    assert all(prediction in ['Positive', 'Negative'] for prediction in predictions), 'The function should return only Positive or Negative.'
    
    print('All tests passed.')

test_classify_movie_reviews()