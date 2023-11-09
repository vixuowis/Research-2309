# function_import --------------------

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# function_code --------------------

def load_pong_model(repo_id: str, filename: str) -> PPO:
    """
    Load a pre-trained PPO model for the PongNoFrameskip-v4 game from the Hugging Face model hub.

    Args:
        repo_id (str): The repository ID where the pre-trained model is stored.
        filename (str): The name of the model file.

    Returns:
        PPO: The loaded PPO model.
    """
    model = PPO.load_from_hub(repo_id, filename)
    return model

# test_function_code --------------------

def test_load_pong_model():
    """
    Test the load_pong_model function.
    """
    model = load_pong_model('sb3/ppo-PongNoFrameskip-v4', 'model.zip')
    env = DummyVecEnv(['PongNoFrameskip-v4'])
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    assert mean_reward >= 21.0 - std_reward, 'Model performance is not as expected.'

# call_test_function_code --------------------

test_load_pong_model()