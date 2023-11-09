def test_extract_features_russian_text():
    """
    This function tests the 'extract_features_russian_text' function.
    """
    # Define a test text in Russian
    test_text = 'Это тестовый текст на русском языке'
    
    # Call the 'extract_features_russian_text' function with the test text
    features = extract_features_russian_text(test_text)
    
    # Assert that the returned object is a torch.Tensor
    assert isinstance(features, torch.Tensor), 'The returned object is not a torch.Tensor.'
    
    # Assert that the shape of the returned tensor is as expected
    assert features.shape[0] == 1, 'The shape of the returned tensor is not as expected.'

test_extract_features_russian_text()