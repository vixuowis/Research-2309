def test_load_graphormer_model():
    """
    This function tests the 'load_graphormer_model' function by loading the model and checking its type.
    """
    # Call the function to load the model
    model = load_graphormer_model()
    # Check if the loaded model is of the correct type
    assert isinstance(model, AutoModel), 'Model loading failed!'

test_load_graphormer_model()