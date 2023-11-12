# function_import --------------------

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# function_code --------------------

def load_and_test_model(checkpoint_name: str, zip_name: str, env_name: str, n_eval_episodes: int):
    """
    Load a trained PPO model from Hugging Face Model Hub and evaluate its performance.

    Args:
        checkpoint_name (str): The name of the checkpoint to load.
        zip_name (str): The name of the zip file containing the checkpoint.
        env_name (str): The name of the environment to test the model on.
        n_eval_episodes (int): The number of episodes to evaluate the model on.

    Returns:
        mean_reward (float): The mean reward obtained over the evaluation episodes.
        std_reward (float): The standard deviation of the reward obtained over the evaluation episodes.
    """
    checkpoint = load_from_hub(checkpoint_name, zip_name)
    model = PPO.load(checkpoint)
    env = make_vec_env(env_name, n_envs=1)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    return mean_reward, std_reward

# test_function_code --------------------

def test_load_and_test_model():
    """
    Test the load_and_test_model function.
    """
    mean_reward, std_reward = load_and_test_model('araffin/ppo-LunarLander-v2', 'ppo-LunarLander-v2.zip', 'LunarLander-v2', 20)
    assert isinstance(mean_reward, float)
    assert isinstance(std_reward, float)
    assert mean_reward > 0
    assert std_reward >= 0
    return 'All Tests Passed'

# call_test_function_code --------------------

test_load_and_test_model()