# function_import --------------------

from mlagents.trainers import TrainerFactory, load_config
from mlagents_envs.environment import UnityEnvironment

# function_code --------------------

def load_and_train_model(repo_id: str, local_dir: str, config_path: str, run_id: str):
    """
    Load a trained POCA model from Hugging Face and train it in a custom SoccerTwos environment.

    Args:
        repo_id (str): The repository id of the trained model on Hugging Face.
        local_dir (str): The local directory where the model will be downloaded.
        config_path (str): The path to the configuration file for the SoccerTwos environment and the poca model.
        run_id (str): The run id for the training session.

    Returns:
        None
    """
    # Load the model from Hugging Face
    TrainerFactory.load_model(repo_id, local_dir)

    # Load the configuration file
    config = load_config(config_path)

    # Create a Unity environment
    env = UnityEnvironment(file_name=None, base_port=5005)

    # Train the model
    TrainerFactory(env, run_id, config).train()

# test_function_code --------------------

def test_load_and_train_model():
    """
    Test the load_and_train_model function.

    This function does not return anything but will raise an error if the function is not working correctly.
    """
    # Define the test parameters
    repo_id = '0xid/poca-SoccerTwos'
    local_dir = './downloads'
    config_path = './config.yaml'
    run_id = 'test_run'

    # Call the function with the test parameters
    load_and_train_model(repo_id, local_dir, config_path, run_id)

# call_test_function_code --------------------

test_load_and_train_model()