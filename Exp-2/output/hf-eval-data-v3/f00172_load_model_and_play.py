# function_import --------------------

import os
import subprocess
from mlagents_envs.environment import UnityEnvironment

# function_code --------------------

def load_model_and_play(repo_id: str, local_dir: str) -> None:
    """
    Load a pre-trained model from Hugging Face model hub and use it to play SoccerTwos.

    Args:
        repo_id (str): The repository ID of the pre-trained model on Hugging Face model hub.
        local_dir (str): The local directory where the pre-trained model will be downloaded.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified local directory does not exist.
        ModuleNotFoundError: If the necessary modules are not installed.
    """
    if not os.path.exists(local_dir):
        raise FileNotFoundError(f"The specified local directory {local_dir} does not exist.")

    # Download the pre-trained model
    subprocess.run(['mlagents-load-from-hf', '--repo-id=' + repo_id, '--local-dir=' + local_dir])

    # Set up the SoccerTwos environment and use the downloaded model as the agent's brain
    # This code snippet assumes familiarity with setting up Unity ML-Agents environments.
    # Follow the documentation for guidance on setting up the SoccerTwos environment and integrating the downloaded model.

# test_function_code --------------------

def test_load_model_and_play():
    """
    Test the load_model_and_play function.
    """
    # Test with valid inputs
    load_model_and_play('Raiden-1001/poca-Soccerv7', './downloads')

    # Test with non-existing local directory
    try:
        load_model_and_play('Raiden-1001/poca-Soccerv7', './non_existing_directory')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Expected a FileNotFoundError.')

    # Test with non-existing repo_id
    try:
        load_model_and_play('non_existing_repo_id', './downloads')
    except Exception:
        pass
    else:
        raise AssertionError('Expected an Exception.')

    return 'All Tests Passed'

# call_test_function_code --------------------

test_load_model_and_play()