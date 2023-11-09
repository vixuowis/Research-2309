def test_load_pretrained_graphormer():
    """
    This function tests the load_pretrained_graphormer function.
    It asserts that the function returns an instance of the correct class.
    """
    model = load_pretrained_graphormer()
    assert isinstance(model, AutoModel), 'Model loading failed'

test_load_pretrained_graphormer()