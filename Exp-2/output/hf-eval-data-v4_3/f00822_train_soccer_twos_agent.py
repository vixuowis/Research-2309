# requirements_file --------------------

import subprocess

requirements = ["subprocess.run", "pyyaml"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import subprocess
import yaml
import os

# function_code --------------------

def train_soccer_twos_agent(configuration_file, run_id, resume=False):
    """
    Train the agent to play SoccerTwos using a pretrained model.

    Args:
        configuration_file (str): The path to the YAML configuration file.
        run_id (str): Unique identifier for the training run.
        resume (bool): Flag to resume training from a checkpoint. Default is False.

    Returns:
        str: The output from the Unity ML-Agent training command.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        subprocess.CalledProcessError: If there is an error during the training command execution.
    """
    if not os.path.exists(configuration_file):
        raise FileNotFoundError(f"Configuration file not found: {configuration_file}")

    command = [
        'mlagents-learn',
        configuration_file,
        '--run-id', run_id
    ]

    if resume:
        command.append('--resume')

    return subprocess.check_output(command, text=True)

# test_function_code --------------------

def test_train_soccer_twos_agent():
    print("Testing started.")
    # Mock object to simulate 'os.path.exists' method
    class MockOSPath:
        @staticmethod
        def exists(file_path):
            return file_path == "valid_configuration_file.yaml"

    # Inject mock object
    train_soccer_twos_agent.os = MockOSPath

    # Test case 1: Valid configuration file with 'resume' set to True
    print("Testing case [1/3] started.")
    assert train_soccer_twos_agent("valid_configuration_file.yaml", "test_run", True) == "Training completed", f"Test case [1/3] failed: Invalid output when 'resume' is True with valid file."

    # Test case 2: Valid configuration file with 'resume' set to False
    print("Testing case [2/3] started.")
    assert train_soccer_twos_agent("valid_configuration_file.yaml", "test_run", False) == "Training completed", f"Test case [2/3] failed: Invalid output when 'resume' is False with valid file."

    # Test case 3: Invalid configuration file
    print("Testing case [3/3] started.")
    try:
        train_soccer_twos_agent("invalid.yaml", "test_run", False)
        assert False, "Test case [3/3] failed: FileNotFoundError not raised with invalid file."
    except FileNotFoundError:
        assert True
    print("Testing finished.")


# call_test_function_line --------------------

test_train_soccer_twos_agent()