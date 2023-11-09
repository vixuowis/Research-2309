def test_load_and_evaluate_model():
    """
    Test the load_and_evaluate_model function.
    """
    checkpoint = 'araffin/dqn-LunarLander-v2'
    kwargs = dict(target_update_interval=30)
    env_name = 'LunarLander-v2'
    n_eval_episodes = 20
    mean_reward, std_reward = load_and_evaluate_model(checkpoint, kwargs, env_name, n_eval_episodes)
    assert isinstance(mean_reward, float), 'Mean reward should be a float.'
    assert isinstance(std_reward, float), 'Standard deviation should be a float.'