# function_import --------------------

import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from rl_zoo3 import load_from_hub

# function_code --------------------

def evaluate_dqn_cartpole(model_filename: str, n_eval_episodes: int = 10):
    """
    Evaluate the performance of a pre-trained DQN agent on the CartPole-v1 environment.

    Args:
        model_filename (str): The filename of the pre-trained model.
        n_eval_episodes (int, optional): The number of evaluation episodes. Defaults to 10.

    Returns:
        tuple: A tuple containing the mean reward and the standard deviation.
    """
    os.environ["SB3-HUB_REPO_ID"] = "sb3/dqn-CartPole-v1"
    model = load_from_hub(repo_id="sb3/dqn-CartPole-v1", filename=model_filename)

    env = gym.make("CartPole-v1")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)

    return mean_reward, std_reward

# test_function_code --------------------

def test_evaluate_dqn_cartpole():
    """
    Test the function evaluate_dqn_cartpole.
    """
    model_filename = "{MODEL FILENAME}.zip"
    mean_reward, std_reward = evaluate_dqn_cartpole(model_filename)

    assert isinstance(mean_reward, float)
    assert isinstance(std_reward, float)
    assert mean_reward > 0
    assert std_reward >= 0

# call_test_function_code --------------------

test_evaluate_dqn_cartpole()