# requirements_file --------------------

import subprocess

requirements = ["rl_zoo3", "sb3", "sb3_contrib"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from rl_zoo3.load_from_hub import load_from_hub

# function_code --------------------

def load_pretrained_dqn_model(model_filename: str):
    """
    Load a pre-trained Deep Q-Network (DQN) model for the MountainCar-v0 environment.

    Args:
        model_filename (str): The filename of the pretrained model file, including extension.

    Returns:
        The loaded DQN model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    try:
        model = load_from_hub(repo_id='sb3/dqn-MountainCar-v0', filename=model_filename)
        return model
    except FileNotFoundError:
        raise FileNotFoundError("Model file not found.")

# test_function_code --------------------

def test_load_pretrained_dqn_model():
    print("Testing started.")

    # Test case 1: Valid model file
    print("Testing case [1/3] started.")
    valid_model = load_pretrained_dqn_model('valid_model.zip')
    assert valid_model is not None, 'Test case [1/3] failed: Model did not load properly.'

    # Test case 2: Invalid model file
    print("Testing case [2/3] started.")
    try:
        load_pretrained_dqn_model('invalid_model.zip')
        assert False, 'Test case [2/3] failed: FileNotFoundError was not raised for an invalid model file.'
    except FileNotFoundError:
        assert True

    # Test case 3: Missing file extension
    print("Testing case [3/3] started.")
    try:
        load_pretrained_dqn_model('valid_model')
        assert False, 'Test case [3/3] failed: FileNotFoundError was not raised for a missing file extension.'
    except FileNotFoundError:
        assert True

    print("Testing finished.")

# call_test_function_line --------------------

test_load_pretrained_dqn_model()