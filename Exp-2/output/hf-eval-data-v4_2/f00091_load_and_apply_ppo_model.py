# requirements_file --------------------

!pip install -U rl-zoo3 stable-baselines3 stable-baselines3-contrib

# function_import --------------------

import rl_zoo3
from stable_baselines3 import PPO

# function_code --------------------

def load_and_apply_ppo_model(repo_id, filename):
    """
    Loads a pre-trained PPO model from the RL Zoo and applies it to the CartPole environment
    for the purpose of optimizing robotic arm warehouse tasks.

    Args:
        repo_id (str): The repository ID where the model is stored.
        filename (str): The filename of the zip file containing the trained model.

    Returns:
        PPO: The loaded PPO model ready to be applied to the environment.

    Raises:
        FileNotFoundError: If the filename does not exist in the specified repository.
    """
    try:
        model = rl_zoo3.load_from_hub(repo_id=repo_id, filename=filename)
    except FileNotFoundError as e:
        raise e
    return model

# test_function_code --------------------

def test_load_and_apply_ppo_model():
    print("Testing started.")
    repo_id = 'HumanCompatibleAI/ppo-seals-CartPole-v0'
    filename = 'pretrained_model.zip'  # Replace with actual model filename
    
    # Test case 1: Check if the model loads correctly
    print("Testing case [1/1] started.")
    try:
        model = load_and_apply_ppo_model(repo_id, filename)
        assert isinstance(model, PPO), "Test case [1/1] failed: The loaded model should be an instance of PPO."
    except FileNotFoundError as e:
        assert False, f"Test case [1/1] failed: FileNotFoundError raised with message {e}."
    print("Testing finished.")

# call_test_function_line --------------------

test_load_and_apply_ppo_model()