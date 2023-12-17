# requirements_file --------------------

!pip install -U huggingface_sb3 stable_baselines3

# function_import --------------------

from huggingface_sb3 import load_from_hub
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# function_code --------------------

def load_and_evaluate_model(checkpoint='araffin/dqn-LunarLander-v2', zip_filename='dqn-LunarLander-v2.zip', target_update_interval=30, n_eval_episodes=20):
    # Load the pre-trained model checkpoint
    checkpoint_path = load_from_hub(checkpoint, zip_filename)
    # Specify any additional arguments for the DQN algorithm
    kwargs = dict(target_update_interval=target_update_interval)
    # Load the DQN model with checkpoint and kwargs
    model = DQN.load(checkpoint_path, **kwargs)
    # Create the LunarLander-v2 environment
    env = make_vec_env('LunarLander-v2', n_envs=1)
    # Evaluate the model's performance and print the results
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes, deterministic=True)
    print(f'Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}')
    return mean_reward, std_reward

# test_function_code --------------------

def test_load_and_evaluate_model():
    print('Testing load_and_evaluate_model function.')
    mean_reward, std_reward = load_and_evaluate_model()
    expected_mean_reward = 280.22
    reward_threshold = 10  # Define threshold for variation in mean reward
    assert abs(mean_reward - expected_mean_reward) < reward_threshold, f'Test failed: Mean reward outside acceptable range.'
    print('Test passed: Mean reward within acceptable range.')