# function_import --------------------

from stable_baselines3 import DQN
from rl_zoo3.load_from_hub import load_from_hub

# function_code --------------------

def load_model(repo_id: str, filename: str):
    '''
    Load a pre-trained model from the RL Zoo.

    Args:
        repo_id (str): The repository ID of the model. For example, 'sb3/dqn-MountainCar-v0'.
        filename (str): The filename of the model. This should be a .zip file.

    Returns:
        A loaded model.
    '''
    model = load_from_hub(repo_id=repo_id, filename=filename)
    return model

# test_function_code --------------------

def test_load_model():
    '''
    Test the load_model function.
    '''
    model = load_model(repo_id='sb3/dqn-MountainCar-v0', filename='model.zip')
    assert isinstance(model, DQN), 'Model is not an instance of DQN.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_load_model()