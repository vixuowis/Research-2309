# function_import --------------------

from mlagents_envs.environment import UnityEnvironment
from mlagents.trainers.trainer_util import load_config
from mlagents.trainers import TrainerFactory

# function_code --------------------

def train_soccer_agent(config_path: str, run_id: str):
    """
    Trains a soccer agent using the ML-Agents library.

    Args:
        config_path (str): The path to the configuration YAML file.
        run_id (str): The identifier for the training run.

    Returns:
        None
    """
    # Load the configuration
    config = load_config(config_path)
    # Create the Unity environment
    env = UnityEnvironment(file_name='SoccerTwos')
    # Create the trainer factory
    factory = TrainerFactory(config)
    # Train the agent
    for episode in range(10000):
        env.reset()
        for agent in env.get_agent_groups():
            decision_steps, terminal_steps = env.get_steps(agent)
            factory.train(decision_steps, terminal_steps)
        if episode % 1000 == 0:
            print(f'Episode {episode} completed.')

# test_function_code --------------------

def test_train_soccer_agent():
    """
    Tests the train_soccer_agent function.

    Returns:
        None
    """
    # Define a dummy config path and run id
    config_path = 'dummy_config.yaml'
    run_id = 'test_run'
    # Call the function
    train_soccer_agent(config_path, run_id)

# call_test_function_code --------------------

test_train_soccer_agent()