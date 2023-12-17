# requirements_file --------------------

!pip install -U huggingface_sb3 stable_baselines3

# function_import --------------------

from huggingface_sb3 import load_from_hub
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# function_code --------------------

def evaluate_lunar_lander_model(model_name: str, kwargs: dict = None) -> tuple:
    """
    Load a pretrained DQN model and evaluate its performance on the LunarLander-v2 environment.

    Args:
        model_name (str): The name of the pretrained model to load.
        kwargs (dict, optional): Additional keyword arguments for loading the model.

    Returns:
        tuple: A tuple containing the mean reward and the standard deviation.

    Raises:
        FileNotFoundError: If the checkpoint file is not found.
        ValueError: If the environment name is not correct.
    """
    if kwargs is None:
        kwargs = {}
    checkpoint = load_from_hub(model_name, model_name + '.zip')
    model = DQN.load(checkpoint, **kwargs)
    env = make_vec_env('LunarLander-v2', n_envs=1)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
    return mean_reward, std_reward

# test_function_code --------------------

def test_evaluate_lunar_lander_model():
    print("Testing started.")
    
    # Test case 1: Check if the function returns a tuple
    print("Testing case [1/1] started.")
    mean_reward, std_reward = evaluate_lunar_lander_model('araffin/dqn-LunarLander-v2', {'target_update_interval': 30})
    assert isinstance(mean_reward, float) and isinstance(std_reward, float), f"Test case [1/1] failed: expected float return types, got {type(mean_reward)} and {type(std_reward)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_evaluate_lunar_lander_model()