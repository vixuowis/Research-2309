def test_extract_features():
    """
    This function tests the 'extract_features' function by using a sample source code text.
    The test is successful if the function does not raise any exceptions and returns a tensor.
    """
    source_code_text = '/* Sample source code text */'
    feature_matrix = extract_features(source_code_text)
    assert isinstance(feature_matrix, torch.Tensor), 'The function should return a tensor.'
    assert feature_matrix.shape[0] > 0, 'The tensor should not be empty.'

test_extract_features()