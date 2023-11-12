# function_import --------------------

import os
import subprocess
from mlagents_envs.environment import UnityEnvironment

# function_code --------------------

def load_and_train_agent(repo_id: str, local_dir: str, config_file_path: str, run_id: str):
    """
    Load a pre-trained model from Hugging Face Model Hub and resume training using Unity ML-Agents.

    Args:
        repo_id (str): The repository id of the pre-trained model on Hugging Face Model Hub.
        local_dir (str): The local directory where the downloaded model files will be stored.
        config_file_path (str): The path to the configuration file for setting up and training the agent.
        run_id (str): A unique identifier for the training run.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified local directory or configuration file does not exist.
        subprocess.CalledProcessError: If there is an error executing the mlagents-load-from-hf or mlagents-learn commands.
    """
    if not os.path.exists(local_dir):
        raise FileNotFoundError(f"The specified local directory {local_dir} does not exist.")
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"The specified configuration file {config_file_path} does not exist.")

    # Download the pre-trained model from Hugging Face Model Hub
    subprocess.check_call(['mlagents-load-from-hf', '--repo-id', repo_id, '--local-dir', local_dir])

    # Resume training the agent
    subprocess.check_call(['mlagents-learn', config_file_path, '--run-id', run_id, '--resume'])

# test_function_code --------------------

def test_load_and_train_agent():
    """
    Test the load_and_train_agent function.
    """
    # Test with valid inputs
    try:
        load_and_train_agent('Raiden-1001/poca-Soccerv7.1', './downloads', './config.yaml', 'test_run')
    except Exception as e:
        assert False, f"Unexpected error with valid inputs: {e}"

    # Test with non-existent local directory
    try:
        load_and_train_agent('Raiden-1001/poca-Soccerv7.1', './non_existent_directory', './config.yaml', 'test_run')
    except FileNotFoundError:
        pass
    except Exception as e:
        assert False, f"Unexpected error with non-existent local directory: {e}"

    # Test with non-existent configuration file
    try:
        load_and_train_agent('Raiden-1001/poca-Soccerv7.1', './downloads', './non_existent_config.yaml', 'test_run')
    except FileNotFoundError:
        pass
    except Exception as e:
        assert False, f"Unexpected error with non-existent configuration file: {e}"

    return 'All Tests Passed'

# call_test_function_code --------------------

if __name__ == '__main__':
    print(test_load_and_train_agent())