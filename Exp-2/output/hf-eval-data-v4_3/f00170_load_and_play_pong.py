# requirements_file --------------------

import subprocess

requirements = ["stable-baselines3", "gym"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym

# function_code --------------------

def load_and_play_pong(model_filename: str) -> float:
    '''
    Load a pre-trained PPO model and evaluate its performance on PongNoFrameskip-v4.

    Args:
        model_filename (str): The filename of the pre-trained PPO model zip file.

    Returns:
        float: The mean reward obtained by the model on the environment.

    Raises:
        FileNotFoundError: If the model file is not found.
    '''
    # Load the environment and wrap it
    env = gym.make('PongNoFrameskip-v4')
    env = DummyVecEnv([lambda: env])

    # Load the pre-trained model
    model = PPO.load(model_filename, env)

    # Evaluate the policy
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    # Close the environment
    env.close()

    return mean_reward

# test_function_code --------------------

def test_load_and_play_pong():
    print("Testing started.")
    # Mock data for testing since we cannot fetch the actual environment in a test case
    model_filename = 'test_pong_model.zip'

    # Testing case 1: Ensure the function returns a float
    print("Testing case [1/1] started.")
    reward = load_and_play_pong(model_filename)
    assert isinstance(reward, float), f"Test case [1/1] failed: Expected a float, got {type(reward)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_load_and_play_pong()