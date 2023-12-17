# requirements_file --------------------

!pip install -U ml-agents pyyaml

# function_import --------------------

from mlagents.trainers import PPOTrainer
import yaml
import os

# function_code --------------------

def train_soccer_twos_agent(configuration_yaml, run_id):
    """
    Train the SoccerTwos agent using the Unity ML-Agents library.

    Args:
    configuration_yaml (str): Path to the configuration.yaml file containing training parameters.
    run_id (str): A unique identifier for the training run.

    Returns:
    PPOTrainer: The trained PPOTrainer instance.
    """
    # Load the configuration for the training
    with open(configuration_yaml, 'r') as file:
        config = yaml.safe_load(file)

    # Set the run_id in the configuration
    config['run_id'] = run_id

    # Initialize the trainer
    trainer = PPOTrainer(config, run_id)

    # Start the training process
    trainer.train()

    # Return the trained trainer
    return trainer

# test_function_code --------------------

def test_train_soccer_twos_agent():
    print("Testing train_soccer_twos_agent started.")

    # Testing with a mock configuration file path and run_id
    configuration_yaml = 'path/to/mock_configuration.yaml'
    run_id = 'test_run_001'

    # Mock the training process (assuming it's successful)
    trainer = train_soccer_twos_agent(configuration_yaml, run_id)

    # Test case: Check if a trainer is returned
    assert isinstance(trainer, PPOTrainer), f"Test case failed: Expected a PPOTrainer instance, got {type(trainer)}"

    print("Testing train_soccer_twos_agent finished.")

# Run the test
try:
    test_train_soccer_twos_agent()
except AssertionError as e:
    print(e)