def test_load_depth_estimation_model():
    """
    Test the load_depth_estimation_model function.
    """
    model_name = 'sayakpaul/glpn-nyu-finetuned-diode-221122-082237'
    model = load_depth_estimation_model(model_name)
    assert model is not None, 'Model loading failed.'
    assert isinstance(model, AutoModel), 'The loaded model is not an instance of AutoModel.'

test_load_depth_estimation_model()