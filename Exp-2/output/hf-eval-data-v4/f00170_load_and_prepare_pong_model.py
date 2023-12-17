# requirements_file --------------------

!pip install -U stable_baselines3 rl_zoo3

# function_import --------------------

from stable_baselines3 import PPO
from rl_zoo3 import load_from_hub

# function_code --------------------

def load_and_prepare_model(model_filename):
    '''
    Loads and prepares a trained PPO model from Hugging Face Hub for the Pong No Frameskip-v4 environment.
    
    Args:
        model_filename (str): The filename of the model to load (should end with .zip).
    
    Returns:
        model: The loaded PPO model ready for use.
    '''
    # Load the model from Hugging Face Hub
    model = load_from_hub(repo_id='sb3/ppo-PongNoFrameskip-v4', filename=model_filename)
    
    # The model is now ready and can be returned or used directly
    return model

# test_function_code --------------------

def test_load_and_prepare_model():
    print('Testing load_and_prepare_model function.')

    # Replace '{MODEL FILENAME}' with an actual model file name
    model_filename = 'pretrained_model_file.zip'
    model = load_and_prepare_model(model_filename)

    assert model is not None, 'Model was not loaded.'
    assert isinstance(model, PPO), 'Loaded model is not a PPO instance.'

    print('Test passed successfully!')

# Run the test
if __name__ == '__main__':
    test_load_and_prepare_model()