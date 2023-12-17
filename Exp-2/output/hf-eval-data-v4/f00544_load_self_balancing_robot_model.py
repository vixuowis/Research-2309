# requirements_file --------------------

!pip install -U rl_zoo3 stable-baselines3 stable-baselines3-contrib

# function_import --------------------

from rl_zoo3 import load_from_hub
from stable_baselines3 import PPO

# function_code --------------------

def load_self_balancing_robot_model(model_filename):
    """
    Load the pre-trained PPO model for a two-wheeled self-balancing robot.

    Parameters:
        model_filename (str): The name of the model file to load.

    Returns:
        The loaded PPO model.
    """
    repo_id = 'sb3/ppo-CartPole-v1'
    model = load_from_hub(repo_id=repo_id, filename=model_filename)
    return model

# test_function_code --------------------

def test_load_self_balancing_robot_model():
    print("Testing load_self_balancing_robot_model function.")
    model_filename = 'pretrained_model.zip'

    # Test loading the model
    print("Testing model loading started.")
    model = load_self_balancing_robot_model(model_filename)
    assert isinstance(model, PPO), f"Failed to load a PPO model: {type(model)}"

    print("Testing model loading finished successfully.")

# Run the test function
test_load_self_balancing_robot_model()