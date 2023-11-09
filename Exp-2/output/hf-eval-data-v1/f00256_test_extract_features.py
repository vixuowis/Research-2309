def test_extract_features():
    # Test the extract_features function
    # Define a sample Korean text
    sample_text = '이것은 테스트 문장입니다.'
    # Call the function with the sample text
    features = extract_features(sample_text)
    # Assert that the function returns a tensor
    assert isinstance(features, torch.Tensor), 'The function should return a tensor.'
    # Assert that the tensor is not empty
    assert features.size() != 0, 'The function should return a non-empty tensor.'

test_extract_features()