# function_import --------------------

import os
from mlagents_envs.environment import UnityEnvironment

# function_code --------------------

def load_and_train_soccer_agent(repo_id: str, local_dir: str, config_file_path: str, run_id: str):
    """
    Load a pre-trained model from Hugging Face and train a soccer agent using the Unity ML-Agents library.

    Args:
        repo_id (str): The id of the repository where the pre-trained model is stored.
        local_dir (str): The local directory where the pre-trained model will be downloaded.
        config_file_path (str): The path to the configuration file for the agent and its environment.
        run_id (str): The id for the training run.

    Returns:
        None

    Raises:
        ModuleNotFoundError: If the 'mlagents_envs' module is not found.
    """
    # Download the pre-trained model
    os.system(f'mlagents-load-from-hf --repo-id={repo_id} --local-dir={local_dir}')
    # Train the agent using the custom configuration file
    os.system(f'mlagents-learn {config_file_path} --run-id={run_id} --resume')

# test_function_code --------------------

def test_load_and_train_soccer_agent():
    """
    Test the load_and_train_soccer_agent function.
    """
    # Test with valid inputs
    load_and_train_soccer_agent('0xid/poca-SoccerTwos', './downloads', './config.yaml', 'run1')
    assert os.path.exists('./downloads/0xid/poca-SoccerTwos'), 'Model not downloaded correctly'
    # Test with invalid repo_id
    try:
        load_and_train_soccer_agent('invalid_repo_id', './downloads', './config.yaml', 'run2')
    except Exception as e:
        assert isinstance(e, ModuleNotFoundError), 'Exception type mismatch'
    # Test with invalid local_dir
    try:
        load_and_train_soccer_agent('0xid/poca-SoccerTwos', './invalid_dir', './config.yaml', 'run3')
    except Exception as e:
        assert isinstance(e, FileNotFoundError), 'Exception type mismatch'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_load_and_train_soccer_agent()