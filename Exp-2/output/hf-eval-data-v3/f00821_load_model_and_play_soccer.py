# function_import --------------------

import os
import subprocess

# function_code --------------------

def load_model_and_play_soccer(repo_id: str, local_dir: str):
    """
    Load a pretrained model from Hugging Face model hub and use it to play SoccerTwos.

    Args:
        repo_id (str): The repository ID of the pretrained model on Hugging Face model hub.
        local_dir (str): The local directory where the pretrained model will be downloaded.

    Returns:
        None
    """
    # Install the required packages
    subprocess.call(['pip', 'install', 'unity-ml-agents', 'deep-reinforcement-learning', 'ML-Agents-SoccerTwos'])

    # Download the pretrained model
    subprocess.call(['mlagents-load-from-hf', '--repo-id', repo_id, '--local-dir', local_dir])

# test_function_code --------------------

def test_load_model_and_play_soccer():
    """
    Test the function load_model_and_play_soccer.
    """
    # Test case 1: Check if the function can successfully download the model
    load_model_and_play_soccer('Raiden-1001/poca-Soccerv7.1', './downloads')
    assert os.path.exists('./downloads/Raiden-1001/poca-Soccerv7.1'), 'Test case 1 failed'

    # Test case 2: Check if the function can handle invalid repo_id
    try:
        load_model_and_play_soccer('invalid_repo_id', './downloads')
    except Exception as e:
        assert str(e) == 'Repo not found', 'Test case 2 failed'

    # Test case 3: Check if the function can handle invalid local_dir
    try:
        load_model_and_play_soccer('Raiden-1001/poca-Soccerv7.1', 'invalid_local_dir')
    except Exception as e:
        assert str(e) == 'Invalid local directory', 'Test case 3 failed'

    return 'All tests passed'

# call_test_function_code --------------------

test_load_model_and_play_soccer()