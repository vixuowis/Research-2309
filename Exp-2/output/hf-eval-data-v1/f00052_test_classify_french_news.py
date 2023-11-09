def test_classify_french_news():
    '''
    This function tests the classify_french_news function.
    It uses a sample news article and checks if the function returns a dictionary.
    '''
    # Define a sample news article
    sequence = 'L\'équipe de France joue aujourd\'hui au Parc des Princes'
    
    # Call the function with the sample article
    result = classify_french_news(sequence)
    
    # Check if the result is a dictionary
    assert isinstance(result, dict), 'The function should return a dictionary.'
    
    # Check if the dictionary contains the correct keys
    assert set(result.keys()) == set(['sport', 'politique', 'science']), 'The dictionary should contain the categories as keys.'
    
    # Check if the values in the dictionary are probabilities
    assert all(isinstance(value, float) and 0 <= value <= 1 for value in result.values()), 'The values in the dictionary should be probabilities.'

test_classify_french_news()