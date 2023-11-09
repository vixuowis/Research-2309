def test_load_model():
    """
    This function tests the 'load_model' function by loading a model and checking if it is not None.
    """
    model = load_model(repo_id='sb3/dqn-MountainCar-v0', filename='test_model.zip')
    assert model is not None, 'Model loading failed.'