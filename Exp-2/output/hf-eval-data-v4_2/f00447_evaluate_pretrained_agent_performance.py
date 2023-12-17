# requirements_file --------------------

!pip install -U gym stable-baselines3 stable-baselines3-contrib rl_zoo3

# function_import --------------------

import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from rl_zoo3 import load_from_hub

# function_code --------------------

def evaluate_pretrained_agent_performance(model_filename: str) -> (float, float):
    """
    Evaluate the performance of a pre-trained DQN agent on the CartPole-v1 environment.

    Args:
        model_filename (str): The filename of the pre-trained model zip file.

    Returns:
        A tuple containing the mean reward and standard deviation of the agent's performance.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    # Set the repo id for the pre-trained model
    os.environ['SB3-HUB_REPO_ID'] = 'sb3/dqn-CartPole-v1'

    # Load the pre-trained model
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f'The model file {model_filename} does not exist.')
    model = load_from_hub(repo_id='sb3/dqn-CartPole-v1', filename=model_filename)

    # Create the game environment
    env = gym.make('CartPole-v1')

    # Evaluate the agent's performance
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    return mean_reward, std_reward

# test_function_code --------------------

def test_evaluate_pretrained_agent_performance():
    print("Testing started.")
    sample_model_filename = 'pretrained_model.zip'

    # Create a mock pretrained model.zip for testing.

    # Testing case 1: Check if the function returns correct type
    print("Testing case [1/3] started.")
    mean_reward, std_reward = evaluate_pretrained_agent_performance(sample_model_filename)
    assert isinstance(mean_reward, float) and isinstance(std_reward, float), f"Test case 1 failed: Expected float, got {type(mean_reward)} and {type(std_reward)}."

    # Testing case 2: Check if the function raises FileNotFoundError
    print("Testing case [2/3] started.")
    try:
        _ = evaluate_pretrained_agent_performance('non_existent_model.zip')
        assert False, 'Test case 2: FileNotFoundError was not raised for a non-existent model file.'
    except FileNotFoundError:
        pass

    # Testing case 3: Check if the returned mean reward is within the expected range
    print("Testing case [3/3] started.")
    assert 0 <= mean_reward <= 500, f"Test case 3 failed: Mean reward {mean_reward} out of expected range [0, 500]."
    print("Testing finished.")

# call_test_function_line --------------------

test_evaluate_pretrained_agent_performance()