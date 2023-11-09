def test_load_graphormer_model():
    """
    This function tests the 'load_graphormer_model' function by loading the model and checking its type.
    """
    model = load_graphormer_model()
    assert isinstance(model, AutoModel), 'Model loading failed.'