def test_load_pretrained_model():
    """
    This function tests the load_pretrained_model function.
    It asserts that the returned model is not None.
    """
    repo_id = 'sb3/ppo-CartPole-v1'
    filename = 'pretrained_model.zip'
    model = load_pretrained_model(repo_id, filename)
    assert model is not None, 'Model loading failed.'