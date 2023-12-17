# requirements_file --------------------

!pip install -U rl_zoo3 stable-baselines3 stable-baselines3-contrib

# function_import --------------------

import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from rl_zoo3 import load_from_hub

# function_code --------------------

def evaluate_cartpole_dqn_performance(model_filename):
    """
    Evaluate the performance of a DQN model on the CartPole-v1 environment.

    :param model_filename: The filename of the pre-trained DQN model.
    :return: A tuple containing the mean reward and the standard deviation of the reward across episodes.
    """
    os.environ['SB3-HUB_REPO_ID'] = 'sb3/dqn-CartPole-v1'
    model_path = f"{model_filename}.zip"
    model = load_from_hub(repo_id='sb3/dqn-CartPole-v1', filename=model_path)

    env = gym.make("CartPole-v1")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    return mean_reward, std_reward

# test_function_code --------------------

def test_evaluate_cartpole_dqn_performance():
    print("Testing evaluate_cartpole_dqn_performance.")

    # Example model filename
    model_filename = 'example_model'

    # Test case: Evaluate the performance of pretrained DQN model
    mean_reward, std_reward = evaluate_cartpole_dqn_performance(model_filename)

    assert mean_reward >= 0, f"Mean reward is negative, got: {mean_reward}"
    assert std_reward >= 0, f"Standard deviation of reward is negative, got: {std_reward}"

    print(f"Test case passed: Mean reward: {mean_reward}, Std Reward: {std_reward}")

# Run the test
print("Running test_evaluate_cartpole_dqn_performance...")
test_evaluate_cartpole_dqn_performance()