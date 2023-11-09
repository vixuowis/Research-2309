# function_import --------------------

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# function_code --------------------

def load_model_and_play(repo_id: str, local_dir: str):
    '''
    This function downloads a pre-trained model from the Hugging Face model hub and uses it to play a 2v2 soccer game in the SoccerTwos environment.
    
    Args:
    repo_id: str: The repository ID of the pre-trained model on the Hugging Face model hub.
    local_dir: str: The local directory where the downloaded model will be stored.
    
    Returns:
    None
    
    Raises:
    Exception: If there is an error in downloading the model or setting up the SoccerTwos environment.
    '''
    # Download the model
    os.system(f'mlagents-load-from-hf --repo-id={repo_id} --local-dir={local_dir}')
    
    # Set up the SoccerTwos environment and use the downloaded model as the agent's brain
    # This code snippet assumes familiarity with setting up Unity ML-Agents environments.
    # Follow the documentation for guidance on setting up the SoccerTwos environment and integrating the downloaded model.
    
    # Create the Unity environment
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name='SoccerTwos', side_channels=[channel])
    
    # Reset the environment
    env.reset()
    
    # Play the game
    for episode in range(100):
        env.step()

# test_function_code --------------------

def test_load_model_and_play():
    '''
    This function tests the load_model_and_play function by downloading a pre-trained model and using it to play a 2v2 soccer game in the SoccerTwos environment.
    
    Args:
    None
    
    Returns:
    None
    
    Raises:
    Exception: If there is an error in downloading the model or setting up the SoccerTwos environment.
    '''
    # Test the load_model_and_play function
    load_model_and_play('Raiden-1001/poca-Soccerv7', './downloads')

# call_test_function_code --------------------

test_load_model_and_play()