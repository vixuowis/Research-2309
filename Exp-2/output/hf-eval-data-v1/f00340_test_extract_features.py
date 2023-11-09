def test_extract_features():
    '''
    This function tests the 'extract_features' function with a sample biomedical entity name.
    '''
    # Define a sample entity
    entity = 'covid infection'

    # Get the output of the function
    output = extract_features(entity)

    # Assert that the output is not None
    assert output is not None

    # Assert that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Assert that the size of the output is correct
    assert output.size() == (1, 768)

test_extract_features()