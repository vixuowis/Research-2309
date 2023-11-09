# function_import --------------------

from mlagents_envs.environment import UnityEnvironment
from mlagents.trainers.trainer_util import load_config
from mlagents.trainers.trainer_controller import TrainerController

# function_code --------------------

def load_and_train_agent(repo_id: str, local_dir: str, config_file_path: str, run_id: str):
    """
    Load a pre-trained model from Hugging Face Model Hub and resume training in the SoccerTwos environment.

    Args:
        repo_id (str): The repository id of the pre-trained model on Hugging Face Model Hub.
        local_dir (str): The local directory where the downloaded model files will be stored.
        config_file_path (str): The path to the configuration file.
        run_id (str): The unique run id.

    Returns:
        None
    """
    # Load the pre-trained model
    os.system(f'mlagents-load-from-hf --repo-id={repo_id} --local-dir={local_dir}')

    # Load the configuration file
    config = load_config(config_file_path)

    # Create the Unity environment
    env = UnityEnvironment(file_name=None, base_port=5005)

    # Create the trainer controller
    tc = TrainerController(env, run_id, config)

    # Start training
    tc.start_training()

# test_function_code --------------------

def test_load_and_train_agent():
    """
    Test the load_and_train_agent function.
    """
    # Define the test parameters
    repo_id = 'Raiden-1001/poca-Soccerv7.1'
    local_dir = './downloads'
    config_file_path = './config.yaml'
    run_id = 'test_run'

    # Call the function with the test parameters
    load_and_train_agent(repo_id, local_dir, config_file_path, run_id)

    # Check if the model files have been downloaded
    assert os.path.exists(os.path.join(local_dir, repo_id)), 'Model files not downloaded'

    # Check if the training has started
    assert os.path.exists(os.path.join('./models', run_id)), 'Training not started'

# call_test_function_code --------------------

test_load_and_train_agent()