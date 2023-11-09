def test_extract_features():
    """
    This function is used to test the 'extract_features' function.
    It uses a sample Indonesian text and checks if the output is a torch.Tensor.
    """
    sample_text = 'Saya suka makan nasi goreng'
    output = extract_features(sample_text)
    assert isinstance(output, torch.Tensor), 'Output is not a torch.Tensor'