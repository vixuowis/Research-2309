def test_load_graphormer_model():
    """
    This function tests the load_graphormer_model function by loading the model and checking its type.
    """
    # Load the model using the function
    model = load_graphormer_model()
    # Check that the model is of the correct type
    assert isinstance(model, AutoModel), 'Model loading failed'

test_load_graphormer_model()