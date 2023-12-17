# requirements_file --------------------

!pip install -U mlagents

# function_import --------------------

from mlagents.trainers import TrainerFactory, UnityEnvironment, TrainerController
import yaml
import subprocess
import os

# function_code --------------------

def train_soccer_twos_agent(repo_id, local_dir, config_file_path, run_id, resume=False):
    # Load the pre-trained model from the repository
    subprocess.run(['mlagents-load-from-hf', '--repo-id', repo_id, '--local-dir', local_dir], check=True)

    # Ensure the configuration file is present
    if not os.path.isfile(config_file_path):
        raise ValueError(f"Configuration file not found: {config_file_path}")

    # Set up the environment and the trainer controller
    env = UnityEnvironment()
    trainer_factory = TrainerFactory()
    trainer_controller = TrainerController(
        trainer_factory=trainer_factory,
        environment=env
    )

    # Start training
    trainer_controller.start_learning(run_id, resume=resume)

# test_function_code --------------------

def test_train_soccer_twos_agent():
    print("Testing train_soccer_twos_agent function.")

    # Example variables
    repo_id = 'Raiden-1001/poca-SoccerTwosv2'
    local_dir = './downloads'
    config_file_path = 'config/trainer_config.yaml'
    run_id = 'SoccerTwos_v1'

    try:
        # Test the function
        train_soccer_twos_agent(repo_id, local_dir, config_file_path, run_id)
        print("Function executed without errors.")
    except Exception as e:
        print(f"Function raised an exception: {e}")