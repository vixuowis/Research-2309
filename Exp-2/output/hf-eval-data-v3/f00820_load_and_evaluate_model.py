# function_import --------------------

from huggingface_sb3 import load_from_hub
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# function_code --------------------

def load_and_evaluate_model(checkpoint: str, kwargs: dict, env_name: str, n_envs: int, n_eval_episodes: int, deterministic: bool):
    """
    Load a pre-trained model and evaluate its performance.

    Args:
        checkpoint (str): The model checkpoint.
        kwargs (dict): Optional arguments for the model.
        env_name (str): The name of the environment.
        n_envs (int): The number of environments.
        n_eval_episodes (int): The number of evaluation episodes.
        deterministic (bool): Whether to use deterministic actions.

    Returns:
        tuple: Mean reward and standard deviation.
    """
    model = DQN.load(load_from_hub(checkpoint), **kwargs)
    env = make_vec_env(env_name, n_envs=n_envs)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=deterministic)
    return mean_reward, std_reward

# test_function_code --------------------

def test_load_and_evaluate_model():
    """Test the load_and_evaluate_model function."""
    checkpoint = 'araffin/dqn-LunarLander-v2'
    kwargs = dict(target_update_interval=30)
    env_name = 'LunarLander-v2'
    n_envs = 1
    n_eval_episodes = 20
    deterministic = True
    mean_reward, std_reward = load_and_evaluate_model(checkpoint, kwargs, env_name, n_envs, n_eval_episodes, deterministic)
    assert isinstance(mean_reward, float)
    assert isinstance(std_reward, float)
    print('All Tests Passed')

# call_test_function_code --------------------

test_load_and_evaluate_model()