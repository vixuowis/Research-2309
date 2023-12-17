# requirements_file --------------------

!pip install -U rl-zoo3 stable-baselines3 sb3-contrib

# function_import --------------------

from rl_zoo3.load_from_hub import load_from_hub

# function_code --------------------

def load_pretrained_mountain_car_model(model_filename):
    """
    Load a pre-trained DQN model for the MountainCar-v0 environment from Stable Baselines3.

    :param model_filename: str, the filename of the pre-trained model zip file
    :return: the loaded model
    """
    repo_id = 'sb3/dqn-MountainCar-v0'
    # Replace 'model_filename' with the actual .zip filename for your model
    return load_from_hub(repo_id=repo_id, filename=model_filename)

# test_function_code --------------------

def test_load_pretrained_mountain_car_model():
    print("Testing the model loading.")
    # Use an example filename for the test case
    test_model_filename = 'pretrained_mountaincar_model.zip'
    
    # Attempt to load the model
    loaded_model = load_pretrained_mountain_car_model(test_model_filename)
    
    # Test if the model is loaded correctly, this will differ based on the actual library used
    assert loaded_model is not None, f"Failed to load the model with filename: {test_model_filename}"
    print("Test passed.")

# Running the test
test_load_pretrained_mountain_car_model()