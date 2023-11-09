import os
import subprocess

# Function to load and train a reinforcement learning agent for SoccerTwos

def load_and_train_agent(repo_id: str, local_dir: str, config_file_path: str, run_id: str):
    """
    This function loads a pre-trained model from Hugging Face Model Hub and resumes training to improve the agent's performance.
    
    Parameters:
    repo_id (str): The repository id of the pre-trained model on Hugging Face Model Hub.
    local_dir (str): The local directory where the downloaded model files will be stored.
    config_file_path (str): The path to your configuration file.
    run_id (str): A unique run id.
    
    Returns:
    None
    """
    # Load the pre-trained model from Hugging Face Model Hub
    load_command = f'mlagents-load-from-hf --repo-id={repo_id} --local-dir={local_dir}'
    subprocess.run(load_command, shell=True)
    
    # Resume training to improve the agent's performance
    train_command = f'mlagents-learn {config_file_path} --run-id={run_id} --resume'
    subprocess.run(train_command, shell=True)