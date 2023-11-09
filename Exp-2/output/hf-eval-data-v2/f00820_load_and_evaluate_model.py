# function_import --------------------

from huggingface_sb3 import load_from_hub
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# function_code --------------------

def load_and_evaluate_model(checkpoint: str, kwargs: dict, env_name: str, n_eval_episodes: int, deterministic: bool):
    """
    Load a pre-trained model and evaluate its performance.

    Args:
        checkpoint (str): The model checkpoint to load.
        kwargs (dict): Additional arguments to pass to the DQN.load function.
        env_name (str): The name of the environment to create.
        n_eval_episodes (int): The number of episodes to evaluate the model on.
        deterministic (bool): Whether to use deterministic or stochastic actions.

    Returns:
        tuple: A tuple containing the mean reward and the standard deviation.
    """
    model = DQN.load(checkpoint, **kwargs)
    env = make_vec_env(env_name, n_envs=1)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=deterministic)
    return mean_reward, std_reward

# test_function_code --------------------

def test_load_and_evaluate_model():
    """
    Test the load_and_evaluate_model function.
    """
    checkpoint = 'araffin/dqn-LunarLander-v2'
    kwargs = dict(target_update_interval=30)
    env_name = 'LunarLander-v2'
    n_eval_episodes = 20
    deterministic = True
    mean_reward, std_reward = load_and_evaluate_model(checkpoint, kwargs, env_name, n_eval_episodes, deterministic)
    assert isinstance(mean_reward, float)
    assert isinstance(std_reward, float)

# call_test_function_code --------------------

test_load_and_evaluate_model()