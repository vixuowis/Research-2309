# function_import --------------------

import rl_zoo3
from stable_baselines3 import PPO

# function_code --------------------

def load_pretrained_model(repo_id: str, filename: str):
    """
    Load a pre-trained PPO agent from the RL Zoo.

    Args:
        repo_id (str): The repository ID where the pre-trained model is stored.
        filename (str): The name of the file where the pre-trained model is stored.

    Returns:
        The loaded model.
    """
    model = rl_zoo3.load_from_hub(repo_id=repo_id, filename=filename)
    return model

# test_function_code --------------------

def test_load_pretrained_model():
    """
    Test the load_pretrained_model function.
    """
    model = load_pretrained_model(repo_id='HumanCompatibleAI/ppo-seals-CartPole-v0', filename='model.zip')
    assert model is not None, 'Model loading failed.'

# call_test_function_code --------------------

test_load_pretrained_model()