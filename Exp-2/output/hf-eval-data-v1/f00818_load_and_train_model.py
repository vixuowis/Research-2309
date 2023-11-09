import os
import mlagents


def load_and_train_model(repo_id: str, local_dir: str, config_file_path: str, run_id: str):
    """
    Load a trained POCA model from Hugging Face and train it in a custom SoccerTwos environment.

    Args:
        repo_id (str): The repository id of the trained model on Hugging Face.
        local_dir (str): The local directory to store the downloaded model.
        config_file_path (str): The path of the configuration file for the SoccerTwos environment and the poca trained model.
        run_id (str): The run id for the training session.

    Returns:
        None
    """
    # Load the trained model from Hugging Face
    os.system(f'mlagents-load-from-hf --repo-id={repo_id} --local-dir={local_dir}')

    # Train the model in the custom SoccerTwos environment
    os.system(f'mlagents-learn {config_file_path} --run-id={run_id} --resume')