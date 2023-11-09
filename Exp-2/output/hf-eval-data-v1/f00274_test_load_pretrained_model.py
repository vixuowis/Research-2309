def test_load_pretrained_model():
    """
    This function tests the load_pretrained_model function.
    It asserts that the function returns a model of the correct type.
    """
    # Call the function to test
    model = load_pretrained_model()
    # Assert that the returned model is of the correct type
    assert isinstance(model, AutoModel), 'Model loaded is not of the correct type.'