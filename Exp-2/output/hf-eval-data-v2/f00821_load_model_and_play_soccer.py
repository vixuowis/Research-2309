# function_import --------------------

from mlagents_envs.environment import UnityEnvironment
from mlagents.trainers.trainer_util import load_config
from mlagents.trainers.ppo.trainer import PPOTrainer

# function_code --------------------

def load_model_and_play_soccer(repo_id: str, local_dir: str, config_path: str, run_id: str):
    """
    Load a pretrained model from Hugging Face and use it to play SoccerTwos.

    Args:
        repo_id (str): The id of the repository where the pretrained model is stored.
        local_dir (str): The local directory where the model should be downloaded.
        config_path (str): The path to the configuration file.
        run_id (str): The unique identifier for the run.

    Returns:
        None
    """
    # Download the model
    !mlagents-load-from-hf --repo-id=repo_id --local-dir=local_dir

    # Load the environment
    env = UnityEnvironment(file_name='SoccerTwos')

    # Load the configuration
    config = load_config(config_path)

    # Create the trainer
    trainer = PPOTrainer(env, config)

    # Load the model
    trainer.load_model()

    # Play the game
    trainer.play(run_id)

# test_function_code --------------------

def test_load_model_and_play_soccer():
    """
    Test the function load_model_and_play_soccer.

    Returns:
        None
    """
    # Define the parameters
    repo_id = 'Raiden-1001/poca-Soccerv7.1'
    local_dir = './downloads'
    config_path = 'your_configuration_file_path.yaml'
    run_id = 'test_run'

    # Call the function
    load_model_and_play_soccer(repo_id, local_dir, config_path, run_id)

    # Check if the model file exists
    assert os.path.exists(os.path.join(local_dir, 'model.nn'))

# call_test_function_code --------------------

test_load_model_and_play_soccer()