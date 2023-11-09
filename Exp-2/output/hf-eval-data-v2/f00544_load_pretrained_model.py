# function_import --------------------

from stable_baselines3 import PPO
import rl_zoo3

# function_code --------------------

def load_pretrained_model(model_filename):
    """
    Load a pre-trained model from the RL Zoo.

    Args:
        model_filename (str): The filename of the pre-trained model.

    Returns:
        The loaded model.
    """
    ppo = rl_zoo3.load_from_hub(repo_id='sb3/ppo-CartPole-v1', filename=model_filename)
    return ppo

# test_function_code --------------------

def test_load_pretrained_model():
    """
    Test the load_pretrained_model function.
    """
    model_filename = 'test_model.zip'
    model = load_pretrained_model(model_filename)
    assert isinstance(model, PPO), 'Model loading failed.'

# call_test_function_code --------------------

test_load_pretrained_model()