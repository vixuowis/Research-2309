def test_extract_features_russian_text():
    """
    This function tests the 'extract_features_russian_text' function by using a sample Russian text.
    """
    # Define a sample Russian text
    sample_text = "Пример текста на русском языке."
    
    # Extract features from the sample text
    features = extract_features_russian_text(sample_text)
    
    # Check the type of the returned value
    assert isinstance(features, torch.Tensor), "The function should return a torch.Tensor."
    
    # Check the shape of the returned tensor
    assert len(features.shape) == 2, "The returned tensor should have 2 dimensions."
    
    # Check the size of the second dimension of the returned tensor
    assert features.shape[1] == 768, "The size of the second dimension of the returned tensor should be 768."

test_extract_features_russian_text()