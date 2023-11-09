def test_classify_spanish_article():
    '''
    This function tests the classify_spanish_article function.
    '''
    # Define a test Spanish article
    test_article = 'El autor se perfila, a los 50 años de su muerte, como uno de los grandes de su siglo'
    
    # Classify the test article
    predictions = classify_spanish_article(test_article)
    
    # Assert that the predictions are a dictionary
    assert isinstance(predictions, dict), 'The predictions should be a dictionary.'
    
    # Assert that the predictions contain the correct keys
    for label in ['cultura', 'sociedad', 'economia', 'salud', 'deportes']:
        assert label in predictions, f'The predictions should contain the key {label}.'

test_classify_spanish_article()