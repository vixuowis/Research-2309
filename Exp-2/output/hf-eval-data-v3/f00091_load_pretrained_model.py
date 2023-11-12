# function_import --------------------

import rl_zoo3
from stable_baselines3 import PPO

# function_code --------------------

def load_pretrained_model(model_filename):
    """
    Load a pre-trained PPO agent from the RL Zoo.

    Args:
        model_filename (str): The filename of the pre-trained model.

    Returns:
        A PPO agent.
    """
    model = rl_zoo3.load_from_hub(repo_id='HumanCompatibleAI/ppo-seals-CartPole-v0', filename=model_filename)
    return model

# test_function_code --------------------

def test_load_pretrained_model():
    """
    Test the load_pretrained_model function.
    """
    model = load_pretrained_model('model.zip')
    assert isinstance(model, PPO), 'Model loading failed'
    print('All Tests Passed')

# call_test_function_code --------------------

test_load_pretrained_model()