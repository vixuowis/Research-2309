import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from rl_zoo3 import load_from_hub

os.environ['SB3-HUB_REPO_ID'] = 'sb3/dqn-CartPole-v1'


def evaluate_dqn_cartpole(model_filename):
    """
    This function evaluates the performance of a pre-trained DQN agent on the CartPole-v1 environment.
    It uses the Stable Baselines3 library and the RL Zoo.
    
    Parameters:
    model_filename (str): The filename of the pre-trained model.
    
    Returns:
    float: The mean reward of the agent.
    float: The standard deviation of the agent's reward.
    """
    model_path = model_filename + '.zip'
    model = load_from_hub(repo_id='sb3/dqn-CartPole-v1', filename=model_path)

    env = gym.make('CartPole-v1')
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    return mean_reward, std_reward