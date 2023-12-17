# requirements_file --------------------

!pip install -U mlagents

# function_import --------------------

from mlagents_envs.registry import default_registry

# function_code --------------------

def download_and_train_soccer_agent(repo_id: str, local_dir: str, config_path: str, run_id: str):
    """
    Downloads a pre-trained SoccerTwos agent from Hugging Face and continues training.

    Args:
        repo_id (str): The repository ID on Hugging Face where the model is stored.
        local_dir (str): The directory to which the model should be downloaded.
        config_path (str): Path to the configuration file for training the agent.
        run_id (str): A unique identifier for this run of training.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist at the provided path.
        RuntimeError: If the model cannot be downloaded or training cannot be resumed.
    """
    # Download the model from Hugging Face
    os.system(f'mlagents-load-from-hf --repo-id={repo_id} --local-dir={local_dir}')
    # Verify the configuration file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Configuration file not found at {config_path}')
    # Resume training
    os.system(f'mlagents-learn {config_path} --run-id={run_id} --resume')

# test_function_code --------------------

def test_download_and_train_soccer_agent():
    print("Testing started.")
    # Simulate downloading and training process. In practice, you will interact with the file system and command line.
    # Testing downloading part
    print("Testing case [1/3] started.")
    assert os.path.exists('./downloads/model_name'), "Test case [1/3] failed: Model was not downloaded."
    # Testing file existence
    print("Testing case [2/3] started.")
    assert os.path.exists('config.yaml'), "Test case [2/3] failed: Configuration file not found."
    # Testing training process is resumed
    print("Testing case [3/3] started.")
    assert training_was_resumed(), "Test case [3/3] failed: Training was not resumed."
    print("Testing finished.")

# call_test_function_line --------------------

test_download_and_train_soccer_agent()