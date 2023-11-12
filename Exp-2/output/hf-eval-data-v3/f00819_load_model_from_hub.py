# function_import --------------------

from stable_baselines3 import DQN
from stable_baselines3.common.envs import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# function_code --------------------

def load_model_from_hub(repo_id: str, filename: str) -> DQN:
    '''
    Load a pretrained DQN model from Stable-Baselines3 hub.

    Args:
        repo_id (str): The repository ID of the pretrained model on Stable-Baselines3 hub.
        filename (str): The filename of the model to be loaded.

    Returns:
        DQN: The loaded DQN model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    '''
    model = DQN.load(f'{repo_id}/{filename}')
    return model

# test_function_code --------------------

def test_load_model_from_hub():
    '''
    Test the function load_model_from_hub.
    '''
    try:
        model = load_model_from_hub('sb3', 'dqn-MountainCar-v0.zip')
        assert isinstance(model, DQN)
        print('Test passed.')
    except FileNotFoundError:
        print('Model file not found. Test failed.')

# call_test_function_code --------------------

test_load_model_from_hub()