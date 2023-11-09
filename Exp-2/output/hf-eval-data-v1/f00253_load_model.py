from rl_zoo3.load_from_hub import load_from_hub


def load_model(repo_id: str, filename: str):
    """
    This function loads a pre-trained reinforcement learning model from the RL Zoo using the stable-baselines3 library.
    The model is trained on the 'MountainCar-v0' gym environment using the Deep Q-Network (DQN) algorithm.
    
    Args:
    repo_id (str): The repository ID of the model. For the 'MountainCar-v0' environment, this should be 'sb3/dqn-MountainCar-v0'.
    filename (str): The filename of the downloaded model file. This should be a .zip file.
    
    Returns:
    The loaded model.
    """
    model = load_from_hub(repo_id=repo_id, filename=filename)
    return model