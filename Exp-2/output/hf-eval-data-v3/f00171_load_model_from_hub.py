# function_import --------------------

from rl_zoo3 import load_from_hub

# function_code --------------------

def load_model_from_hub(repo_id: str, filename: str):
    """
    Load a pre-trained PPO agent from the RL Zoo.

    Args:
        repo_id (str): The repository id of the pre-trained model.
        filename (str): The filename of the zip file containing the pre-trained model.

    Returns:
        A pre-trained PPO agent.
    """
    model = load_from_hub(repo_id=repo_id, filename=filename)
    return model

# test_function_code --------------------

def test_load_model_from_hub():
    """
    Test the load_model_from_hub function.
    """
    model = load_model_from_hub(repo_id='sb3/ppo-CartPole-v1', filename='pretrained_model.zip')
    assert model is not None, 'Model loading failed'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_load_model_from_hub()