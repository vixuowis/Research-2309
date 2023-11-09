from rl_zoo3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def load_pretrained_ppo_agent(filename):
    '''
    This function loads a pre-trained PPO (Proximal Policy Optimization) agent from the RL Zoo repository.
    The agent is trained to play the CartPole-v0 game.
    
    Args:
    filename (str): The filename of the trained model file.
    
    Returns:
    trained_model: The loaded PPO agent.
    '''
    repo_id = "HumanCompatibleAI/ppo-seals-CartPole-v0"
    model_path = load_from_hub(repo_id, filename=filename)
    alg = PPO
    env = make_vec_env('seals/CartPole-v0', n_envs=1)
    trained_model = alg.load(model_path, env)
    return trained_model