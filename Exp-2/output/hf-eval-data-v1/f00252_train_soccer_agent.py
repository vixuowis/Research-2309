import os
from mlagents.trainers import UnityTrainerException

# Function to train a soccer agent using ML-Agents
# @param config_file: The path to the configuration YAML file
# @param run_id: The id for the training run
# @return: None

def train_soccer_agent(config_file: str, run_id: str) -> None:
    # Check if the configuration file exists
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} does not exist.")
    
    # Check if the ML-Agents library is installed
    try:
        import mlagents
    except ImportError:
        raise ImportError("The ML-Agents library is not installed.")
    
    # Download the pre-trained model
    os.system("mlagents-load-from-hf --repo-id='0xid/poca-SoccerTwos' --local-dir='./downloads'")
    
    # Train the agent using the custom configuration file
    os.system(f"mlagents-learn {config_file} --run-id={run_id} --resume")