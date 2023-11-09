def test_extract_features():
    """
    This function tests the 'extract_features' function.
    It uses a sample text and checks if the output is a torch.Tensor.
    """
    # Sample text
    text = 'This is a sample text.'
    # Extract features
    features = extract_features(text)
    # Check if the output is a torch.Tensor
    assert isinstance(features, torch.Tensor), 'The output should be a torch.Tensor.'

test_extract_features()