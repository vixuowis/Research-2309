# requirements_file --------------------

!pip install -U huggingface_sb3 stable_baselines3

# function_import --------------------

from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# function_code --------------------

def load_and_evaluate_ppo_model(checkpoint_name, env_id, n_eval_episodes=10):
    """
    Loads a pre-trained PPO model from Huggingface model hub using the checkpoint name.
    Evaluates the model on the specified gym environment ID.

    :param checkpoint_name: The name of the checkpoint to load.
    :param env_id: The gym environment ID on which to evaluate the model.
    :param n_eval_episodes: The number of episodes to run for evaluation.
    :return: A tuple containing the mean reward and standard deviation.
    """
    # Load the model from the Huggingface hub
    checkpoint = load_from_hub(checkpoint_name, f'{checkpoint_name}.zip')
    model = PPO.load(checkpoint)

    # Make the environment
    env = make_vec_env(env_id, n_envs=1)

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)

    return mean_reward, std_reward

# test_function_code --------------------

def test_load_and_evaluate_ppo_model():
    print("Testing load_and_evaluate_ppo_model function.")
    # Test variables
    checkpoint_name = 'araffin/ppo-LunarLander-v2'
    env_id = 'LunarLander-v2'
    n_eval_episodes = 5

    # Test case 1: Load and Evaluate
    print("Testing case [1/2] started.")
    mean_reward, std_reward = load_and_evaluate_ppo_model(checkpoint_name, env_id, n_eval_episodes)
    assert mean_reward is not None and std_reward is not None, f"Test case [1/2] failed: Model did not return mean_reward and std_reward."
    print("mean reward: ", mean_reward, ", std reward: ", std_reward)

    # Test case 2: Fail on invalid checkpoint
    print("Testing case [2/2] started.")
    try:
        load_and_evaluate_ppo_model('nonexistent-checkpoint', env_id, n_eval_episodes)
        assert False, "Test case [2/2] failed: Invalid checkpoint did not raise an error."
    except Exception as e:
        print("Expected error: ", e)
    print("Testing finished.")