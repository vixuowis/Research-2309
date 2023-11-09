def test_evaluate_dqn_cartpole():
    """
    This function tests the evaluate_dqn_cartpole function.
    It asserts that the mean reward and standard deviation are not None.
    """
    mean_reward, std_reward = evaluate_dqn_cartpole('MODEL FILENAME')
    assert mean_reward is not None, 'Mean reward should not be None'
    assert std_reward is not None, 'Standard deviation should not be None'

test_evaluate_dqn_cartpole()