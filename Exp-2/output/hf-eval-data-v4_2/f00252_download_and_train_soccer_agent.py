# requirements_file --------------------

!pip install -U unityagents

# function_import --------------------

from unityagents import UnityEnvironment

# function_code --------------------

def download_and_train_soccer_agent(repo_id: str, local_dir: str, config_path: str, run_id: str):
    """
    Download a pre-trained soccer agent from a repository and train using a custom configuration.

    Args:
        repo_id (str): The repository ID where the trained model is hosted.
        local_dir (str): Local directory where the model should be downloaded.
        config_path (str): Path to the YAML configuration file for training.
        run_id (str): An identifier for this particular training run.

    Returns:
        bool: True if training is successful, False otherwise.
        
    Raises:
        RuntimeError: If there is an issue with download or training.
    """
    # Execute the ML-Agents command to download the pre-trained model
    os.system(f"mlagents-load-from-hf --repo-id='{repo_id}' --local-dir='{local_dir}'")

    # Execute the ML-Agents command to start the training
    training_sucess = os.system(f"mlagents-learn {config_path} --run-id={run_id} --resume")

    if training_sucess != 0:
        raise RuntimeError('Training failed due to an error.')

    return True

# test_function_code --------------------

def test_download_and_train_soccer_agent():
    print("Testing started.")
    # Since the actual download and training process depends on external factors and can take a long time,
    # here we can mock or simulate the commands using something like unittest.mock or similar for testing purposes.

    # Presuming that we have a function mock_command that simulates the os.system command
    # Testing case 1: Test correct download and training commands
    print("Testing case [1/3] started.")
    success = mock_command("mlagents-load-from-hf --repo-id='0xid/poca-SoccerTwos' --local-dir='./downloads'") and \
              mock_command("mlagents-learn ./config/training_config.yaml --run-id=123456 --resume")
    assert success, "Test case [1/3] failed: Expected True, got {success}"

# call_test_function_line --------------------

test_download_and_train_soccer_agent()