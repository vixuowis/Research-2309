# function_import --------------------

from stable_baselines3 import DQN
from stable_baselines3.common.envs import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# function_code --------------------

def load_model(repo_id: str, filename: str):
    """
    Load a pre-trained model from the Stable-Baselines3 RL Zoo.

    Args:
        repo_id (str): The repository ID of the pre-trained model. For example, 'sb3/dqn-MountainCar-v0'.
        filename (str): The filename of the downloaded model file. For example, '{MODEL FILENAME}.zip'.

    Returns:
        The loaded model.
    """
    from rl_zoo3.load_from_hub import load_from_hub
    model = load_from_hub(repo_id=repo_id, filename=filename)
    return model

# test_function_code --------------------

def test_load_model():
    """
    Test the load_model function.
    """
    model = load_model('sb3/dqn-MountainCar-v0', 'model.zip')
    assert isinstance(model, DQN), 'Model type should be DQN.'

# call_test_function_code --------------------

test_load_model()