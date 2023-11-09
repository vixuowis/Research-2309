# function_import --------------------

from rl_zoo3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# function_code --------------------

def load_pretrained_ppo_agent(filename: str, repo_id: str = 'HumanCompatibleAI/ppo-seals-CartPole-v0') -> PPO:
    """
    Load a pre-trained PPO (Proximal Policy Optimization) agent from the RL Zoo repository.

    Args:
        filename (str): The filename of the trained model file.
        repo_id (str, optional): The repository ID. Defaults to 'HumanCompatibleAI/ppo-seals-CartPole-v0'.

    Returns:
        PPO: The loaded PPO agent.
    """
    model_path = load_from_hub(repo_id, filename=filename)
    env = make_vec_env('seals/CartPole-v0', n_envs=1)
    return PPO.load(model_path, env)

# test_function_code --------------------

def test_load_pretrained_ppo_agent():
    """
    Test the load_pretrained_ppo_agent function.
    """
    filename = 'test_model.zip'
    agent = load_pretrained_ppo_agent(filename)
    assert isinstance(agent, PPO), 'The loaded agent is not a PPO agent.'

# call_test_function_code --------------------

test_load_pretrained_ppo_agent()