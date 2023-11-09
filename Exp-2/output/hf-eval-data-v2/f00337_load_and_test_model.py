# function_import --------------------

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# function_code --------------------

def load_and_test_model(checkpoint:str, zip_file:str):
    """
    This function loads a trained PPO model from a checkpoint and tests its performance.

    Args:
        checkpoint (str): The name of the checkpoint to load the model from.
        zip_file (str): The name of the zip file containing the model.

    Returns:
        mean_reward (float): The mean reward obtained by the model over 20 evaluation episodes.
        std_reward (float): The standard deviation of the reward obtained by the model over 20 evaluation episodes.
    """
    checkpoint = load_from_hub(checkpoint, zip_file)
    model = PPO.load(checkpoint)
    env = make_vec_env('LunarLander-v2', n_envs=1)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
    return mean_reward, std_reward

# test_function_code --------------------

def test_load_and_test_model():
    """
    This function tests the load_and_test_model function.
    """
    mean_reward, std_reward = load_and_test_model('araffin/ppo-LunarLander-v2', 'ppo-LunarLander-v2.zip')
    assert mean_reward > 250, 'The mean reward is less than expected.'
    assert std_reward < 20, 'The standard deviation of the reward is higher than expected.'

# call_test_function_code --------------------

test_load_and_test_model()