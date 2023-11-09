def test_load_decision_transformer_model():
    """
    This function tests the 'load_decision_transformer_model' function.
    It checks if the returned model is not None.
    """
    # Call the function
    model = load_decision_transformer_model()
    # Check if the model is not None
    assert model is not None, 'Model loading failed!'

test_load_decision_transformer_model()