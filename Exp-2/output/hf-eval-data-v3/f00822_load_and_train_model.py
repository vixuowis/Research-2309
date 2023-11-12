# function_import --------------------

import os
from mlagents.trainers import TrainerFactory, load_config
from mlagents_envs.environment import UnityEnvironment
from mlagents.trainers.trainer_controller import TrainerController

# function_code --------------------

def load_and_train_model(repo_id: str, local_dir: str, config_file_path: str, run_id: str, resume: bool):
    """
    Load a trained model from a repository and train an agent using the model.

    Args:
        repo_id (str): The ID of the repository where the trained model is stored.
        local_dir (str): The local directory where the model will be downloaded.
        config_file_path (str): The path to the configuration file for training the agent.
        run_id (str): The unique ID for the training run.
        resume (bool): Whether to resume training from a saved checkpoint.

    Returns:
        None
    """
    # Load the model from the repository
    os.system(f'mlagents-load-from-hf --repo-id={repo_id} --local-dir={local_dir}')

    # Load the configuration file
    config = load_config(config_file_path)

    # Create the Unity environment
    env = UnityEnvironment()

    # Create the trainer
    trainer_factory = TrainerFactory(config, run_id)

    # Create the trainer controller
    trainer_controller = TrainerController(trainer_factory, run_id, config, resume)

    # Start training
    trainer_controller.start_training()

# test_function_code --------------------

def test_load_and_train_model():
    """
    Test the load_and_train_model function.
    """
    # Test with a valid repo_id, local_dir, config_file_path, run_id, and resume
    assert load_and_train_model('Raiden-1001/poca-SoccerTwosv2', './downloads', './config.yaml', 'run1', True) is None

    # Test with an invalid repo_id
    with pytest.raises(Exception):
        load_and_train_model('invalid_repo_id', './downloads', './config.yaml', 'run1', True)

    # Test with an invalid local_dir
    with pytest.raises(Exception):
        load_and_train_model('Raiden-1001/poca-SoccerTwosv2', './invalid_dir', './config.yaml', 'run1', True)

    # Test with an invalid config_file_path
    with pytest.raises(Exception):
        load_and_train_model('Raiden-1001/poca-SoccerTwosv2', './downloads', './invalid_config.yaml', 'run1', True)

    # Test with an invalid run_id
    with pytest.raises(Exception):
        load_and_train_model('Raiden-1001/poca-SoccerTwosv2', './downloads', './config.yaml', 'invalid_run_id', True)

# call_test_function_code --------------------

test_load_and_train_model()