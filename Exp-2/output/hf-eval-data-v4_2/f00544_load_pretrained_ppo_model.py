# requirements_file --------------------

!pip install -U rl_zoo3 stable-baselines3 stable-baselines3-contrib

# function_import --------------------

import rl_zoo3
from stable_baselines3 import PPO

# function_code --------------------

def load_pretrained_ppo_model(model_filename: str) -> PPO:
    """
    Loads a pre-trained PPO model for a two-wheeled self-balancing robot.

    Args:
        model_filename (str): The filename of the pre-trained model zip file.

    Returns:
        PPO: The pre-trained PPO model.

    Raises:
        ValueError: If the model filename is empty or not provided.
    """
    if not model_filename:
        raise ValueError('Model filename must be provided')
    # Load the pre-trained model using the load_from_hub function
    ppo_model = rl_zoo3.load_from_hub(repo_id='sb3/ppo-CartPole-v1', filename=model_filename)
    return ppo_model


# test_function_code --------------------

def test_load_pretrained_ppo_model():
    print("Testing started.")
    # Test case 1: Valid model filename
    print("Testing case [1/2] started.")
    model = load_pretrained_ppo_model('stable_model.zip')
    assert isinstance(model, PPO), 'Test case [1/2] failed: Model is not of type PPO'

    # Test case 2: Invalid model filename (empty string)
    print("Testing case [2/2] started.")
    try:
        model = load_pretrained_ppo_model('')
    except ValueError as e:
        assert str(e) == 'Model filename must be provided', 'Test case [2/2] failed: ValueError not raised for empty filename'

    print("Testing finished.")


# call_test_function_line --------------------

test_load_pretrained_ppo_model()