# function_import --------------------

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# function_code --------------------

def load_pong_model(model_file):
    '''
    Load the pre-trained PPO model for Pong No Frameskip-v4 game.

    Args:
        model_file (str): The path to the pre-trained model file.

    Returns:
        model: The loaded PPO model.
    '''
    model = PPO.load(model_file)
    return model

# test_function_code --------------------

def test_load_pong_model():
    '''
    Test the load_pong_model function.
    '''
    model = load_pong_model('path_to_model_file')
    assert isinstance(model, PPO), 'Model loading failed.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_load_pong_model()