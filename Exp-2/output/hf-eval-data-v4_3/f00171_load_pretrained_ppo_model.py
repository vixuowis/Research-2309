# requirements_file --------------------

import subprocess

requirements = ["rl_zoo3", "stable-baselines3", "stable-baselines3-contrib"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from rl_zoo3 import load_from_hub

# function_code --------------------


def load_pretrained_ppo_model(repo_id: str, filename: str) -> 'Stable Baselines3 PPO Model':
    """
    Loads a pretrained PPO model for the CartPole environment.

    Args:
        repo_id (str): Repository ID of the pre-trained model in the RL Zoo.
        filename (str): Filename of the pre-trained model zip file.

    Returns:
        A pre-trained PPO model from Stable Baselines3.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
    """
    try:
        model = load_from_hub(repo_id=repo_id, filename=filename)
        return model
    except FileNotFoundError as e:
        raise FileNotFoundError('The model file does not exist.') from e


# test_function_code --------------------


def test_load_pretrained_ppo_model():
    print("Testing started.")
    # Assuming we have a file 'pretrained_model.zip' in the correct repository
    filename = 'pretrained_model.zip'
    repo_id = 'sb3/ppo-CartPole-v1'

    # Test case 1: The model file exists
    print("Testing case [1/3] started.")
    try:
        model = load_pretrained_ppo_model(repo_id, filename)
        assert model is not None, f"Test case [1/3] failed: Model could not be loaded."
    except FileNotFoundError:
        assert False, f"Test case [1/3] failed: The model file does not exist."

    # Test case 2: The model file does not exist
    print("Testing case [2/3] started.")
    wrong_filename = 'missing_model.zip'
    try:
        model = load_pretrained_ppo_model(repo_id, wrong_filename)
        assert False, f"Test case [2/3] failed: Model should not be found."
    except FileNotFoundError:
        assert True

    # Test case 3: Invalid repository ID
    print("Testing case [3/3] started.")
    wrong_repo_id = 'invalid/repo-id'
    try:
        model = load_pretrained_ppo_model(wrong_repo_id, filename)
        assert False, f"Test case [3/3] failed: Model should not be found with invalid repo ID."
    except FileNotFoundError:
        assert True

    print("Testing finished.")


# call_test_function_line --------------------

test_load_pretrained_ppo_model()