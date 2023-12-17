# requirements_file --------------------

!pip install -U huggingface_sb3 stable_baselines3

# function_import --------------------

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# function_code --------------------

def load_and_evaluate_ppo_model(checkpoint_path: str, env_name: str) -> tuple:
    """
    Loads a PPO model from a given checkpoint and evaluates it on a specified environment.

    Args:
        checkpoint_path (str): The path to the model checkpoint file.
        env_name (str): The name of the environment to evaluate the model on.

    Returns:
        tuple: A tuple containing the mean reward and standard deviation.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the environment name is invalid.
    """
    checkpoint = load_from_hub(checkpoint_path)
    model = PPO.load(checkpoint)
    env = make_vec_env(env_name, n_envs=1)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
    return mean_reward, std_reward


# test_function_code --------------------

from pathlib import Path

def test_load_and_evaluate_ppo_model():
    print('Testing started.')
    checkpoint_path = 'araffin/ppo-LunarLander-v2.zip'
    env_name = 'LunarLander-v2'
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f'Checkpoint file {checkpoint_path} does not exist.')

    # Test case: existing environment and valid checkpoint
    print('Testing case [1/3] started.')
    mean_reward, std_reward = load_and_evaluate_ppo_model(checkpoint_path, env_name)
    assert mean_reward is not None and std_reward is not None, f'Test case [1/3] failed: Expected non-None mean and std reward, got {mean_reward}, {std_reward}'

    print('Testing finished.')


# call_test_function_line --------------------

test_load_and_evaluate_ppo_model()