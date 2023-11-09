from rl_zoo3 import load_from_hub


def load_pretrained_model(repo_id: str, filename: str):
    """
    This function loads a pre-trained model from the RL Zoo using the Stable Baselines3 library.
    The model is a PPO agent trained on the CartPole-v1 environment.

    Args:
        repo_id (str): The repository id of the pre-trained model.
        filename (str): The filename of the zip file containing the pre-trained model.

    Returns:
        The loaded model.
    """
    model = load_from_hub(repo_id=repo_id, filename=filename)
    return model