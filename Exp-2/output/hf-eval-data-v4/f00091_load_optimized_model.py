# requirements_file --------------------

!pip install -U rl_zoo3 stable-baselines3 stable-baselines3-contrib

# function_import --------------------

import rl_zoo3
from stable_baselines3 import PPO

# function_code --------------------

def load_optimized_model(model_filename):
    """
    Load a pre-trained PPO model from the Stable Baselines3 Hub.

    :param model_filename: The filename of the pre-trained model to be loaded.
    :return: The loaded pre-trained PPO model.
    """
    # Load the pre-trained model from the specified repository
    model = rl_zoo3.load_from_hub(
        repo_id='HumanCompatibleAI/ppo-seals-CartPole-v0',
        filename=model_filename
    )
    return model

# test_function_code --------------------

def test_load_optimized_model():
    print("Testing load_optimized_model function.")

    # Test case: Loading the pre-trained model
    try:
        model = load_optimized_model('pretrained_model.zip')
        print("Test case passed. Model loaded successfully.")
    except Exception as e:
        print(f"Test case failed: {e}")

# Run the test function
test_load_optimized_model()