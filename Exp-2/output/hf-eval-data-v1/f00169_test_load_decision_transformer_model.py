def test_load_decision_transformer_model():
    """
    This function tests the load_decision_transformer_model function.
    It uses a known model name and checks if the returned model is not None.
    """
    # Known model name
    model_name = 'edbeeching/decision-transformer-gym-walker2d-expert'
    
    # Load the model
    model = load_decision_transformer_model(model_name)
    
    # Check if the model is not None
    assert model is not None, 'Model loading failed.'
    
    print('Test passed.')

test_load_decision_transformer_model()