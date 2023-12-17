# requirements_file --------------------

!pip install -U rl_zoo3 stable-baselines3 stable-baselines3-contrib 

# function_import --------------------

from stable_baselines3 import PPO
from rl_zoo3 import load_from_hub

# function_code --------------------

def improve_game_experience(model_filename):
    """
    Load the pre-trained PPO agent from Stable Baselines3 Hub to improve the game experience.

    Args:
    - model_filename (str): The filename of the pre-trained model zip file.

    Returns:
    - PPO: The loaded, pre-trained PPO agent.
    """
    repo_id = 'sb3/ppo-CartPole-v1'
    model = load_from_hub(repo_id=repo_id, filename=model_filename)
    # Integrate the model into your game project
    return model


# test_function_code --------------------

def test_improve_game_experience():
    print("Testing improve_game_experience function.")
    model = improve_game_experience('pretrained_model.zip')
    assert isinstance(model, PPO), "Loaded model is not a pre-trained PPO agent."
    print("Test passed! The pre-trained PPO agent was loaded successfully.")

# Run the test function
test_improve_game_experience()
