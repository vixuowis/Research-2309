def test_load_pretrained_ppo_agent():
    '''
    This function tests the load_pretrained_ppo_agent function by loading a pre-trained PPO agent.
    It asserts that the loaded agent is not None.
    '''
    filename = "{TEST MODEL FILENAME_HERE}.zip"
    trained_model = load_pretrained_ppo_agent(filename)
    assert trained_model is not None, 'Failed to load the pre-trained PPO agent.'