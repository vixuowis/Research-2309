# requirements_file --------------------

!pip install -U rl_zoo3 stable-baselines3 stable-baselines3-contrib

# function_import --------------------

import os
from rl_zoo3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# function_code --------------------

def load_and_run_cartpole_agent(model_filename, num_episodes=10):
    """
    Load a pre-trained PPO agent and run it for a specified number of episodes on the CartPole-v0 environment.

    Args:
        model_filename (str): Filename of the pre-trained model to load.
        num_episodes (int): Number of episodes to run and evaluate the agent's performance.

    Returns:
        float: The average reward obtained over the number of episodes.
    """
    # Define the repository ID and the model filename
    repo_id = "HumanCompatibleAI/ppo-seals-CartPole-v0"
    model_path = load_from_hub(repo_id, filename=model_filename)

    # Create the CartPole-v0 environment
    env = make_vec_env('seals/CartPole-v0', n_envs=1)

    # Load the pre-trained model
    alg = PPO
    trained_model = alg.load(model_path, env)

    # Evaluate the agent on the environment for num_episodes
    total_reward = 0
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = trained_model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

    # Calculate the average reward
    average_reward = total_reward / num_episodes
    return average_reward

# test_function_code --------------------

def test_load_and_run_cartpole_agent():
    print("Testing started.")
    model_filename = "ppo_cartpole_model.zip"  # A placeholder name for the pre-trained model
    num_episodes = 5  # Testing on a reduced number of episodes for quicker testing

    # Run the test
    print("Testing load_and_run_cartpole_agent started.")
    average_reward = load_and_run_cartpole_agent(model_filename, num_episodes=num_episodes)

    # Test case 1: The function should return a float value
    assert isinstance(average_reward, float), f"Test case failed: the function should return a float value but got {type(average_reward)}."

    # Test case 2: The average reward should be non-negative
    assert average_reward >= 0, f"Test case failed: the average reward should be non-negative but got {average_reward}."

    print("Testing finished.")

# Run the test function
test_load_and_run_cartpole_agent()