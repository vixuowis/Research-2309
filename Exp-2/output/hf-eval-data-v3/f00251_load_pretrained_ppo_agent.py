# function_import --------------------

from rl_zoo3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# function_code --------------------

def load_pretrained_ppo_agent(filename: str, repo_id: str = 'HumanCompatibleAI/ppo-seals-CartPole-v0') -> PPO:
    '''
    Load a pre-trained PPO (Proximal Policy Optimization) agent from the RL Zoo repository.

    Args:
        filename (str): The filename for the trained model file.
        repo_id (str, optional): The repository ID. Defaults to 'HumanCompatibleAI/ppo-seals-CartPole-v0'.

    Returns:
        PPO: The loaded PPO agent.
    '''
    model_path = load_from_hub(repo_id, filename=filename)
    env = make_vec_env('seals/CartPole-v0', n_envs=1)
    trained_model = PPO.load(model_path, env)
    return trained_model

# test_function_code --------------------

def test_load_pretrained_ppo_agent():
    '''
    Test the load_pretrained_ppo_agent function.
    '''
    try:
        agent = load_pretrained_ppo_agent('model.zip')
        assert isinstance(agent, PPO)
        print('Test passed.')
    except Exception as e:
        print('Test failed.\n', e)

# call_test_function_code --------------------

test_load_pretrained_ppo_agent()