# requirements_file --------------------

!pip install -U unity-ml-agents deep-reinforcement-learning

# function_import --------------------

from mlagents.envs import UnityEnvironment
import time
import subprocess

# function_code --------------------

def setup_virtual_soccer_training(config_path, run_id):
    """
    Setup the virtual soccer training environment using a pre-trained model and
    start the training process.

    Args:
    config_path (str): The file path to the custom configuration YAML file.
    run_id (str): An identifier for this run, used for saving models and summaries.
    """
    # Load the pre-trained model
    subprocess.run(['mlagents-load-from-hf', '--repo-id=0xid/poca-SoccerTwos', '--local-dir=./downloads'], check=True)

    # Start the training process
    subprocess.run(['mlagents-learn', config_path, '--run-id=' + run_id, '--resume'], check=True)

    print('Virtual soccer training setup complete!')

# test_function_code --------------------

def test_setup_virtual_soccer_training():
    print("Testing setup_virtual_soccer_training function.")

    # Test case with example configuration path and run id
    try:
        setup_virtual_soccer_training('configurations/example_config.yaml', 'example_run_id')
        print("Test case passed!")
    except subprocess.CalledProcessError as e:
        print(f"Test case failed with return code: {e.returncode}")

# Run the test function
test_setup_virtual_soccer_training()