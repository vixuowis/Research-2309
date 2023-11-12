# function_import --------------------

import rl_zoo3
from stable_baselines3 import PPO

# function_code --------------------

def load_pretrained_model(model_filename):
    """
    Load a pre-trained model from the RL Zoo.

    Args:
        model_filename (str): The filename of the pre-trained model.

    Returns:
        A PPO object representing the pre-trained model.
    """
    ppo = rl_zoo3.load_from_hub(repo_id='sb3/ppo-CartPole-v1', filename=model_filename)
    return ppo

# test_function_code --------------------

def test_load_pretrained_model():
    """
    Test the load_pretrained_model function.
    """
    model_filename = 'test_model.zip'
    ppo = load_pretrained_model(model_filename)
    assert isinstance(ppo, PPO), 'The returned object is not a PPO instance.'
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_load_pretrained_model())