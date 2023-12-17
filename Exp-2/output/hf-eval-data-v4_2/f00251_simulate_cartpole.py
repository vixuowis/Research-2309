# requirements_file --------------------

!pip install -U rl_zoo3 stable-baselines3 stable-baselines3-contrib

# function_import --------------------

from rl_zoo3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# function_code --------------------

def simulate_cartpole(model_filename: str) -> None:
    """
    Simulates CartPole environment using a pre-trained PPO (Proximal Policy Optimization) model

    Args:
        model_filename (str): The filename of the trained PPO model in .zip format.

    Returns:
        None

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    filename = f"{model_filename}.zip"
    repo_id = "HumanCompatibleAI/ppo-seals-CartPole-v0"
    model_path = load_from_hub(repo_id, filename=filename)
    alg = PPO
    env = make_vec_env('seals/CartPole-v0', n_envs=1)
    trained_model = alg.load(model_path, env)
    
    obs = env.reset()
    for _ in range(1000):
        action, _states = trained_model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    env.close()

# test_function_code --------------------

import os

def test_simulate_cartpole():
    print("Testing started.")

    # Assuming 'test_model.zip' is the correct filename for the simulated testing
    model_filename = "test_model"

    # Testing case 1: Check if the file exists
    print("Testing case [1/1] started.")
    assert os.path.exists(model_filename + ".zip"), f"Test case [1/1] failed: {model_filename}.zip does not exist."

    print("Testing finished.")

# call_test_function_line --------------------

test_simulate_cartpole()