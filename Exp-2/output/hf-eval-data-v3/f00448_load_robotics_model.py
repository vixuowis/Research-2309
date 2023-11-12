# function_import --------------------

from transformers import pipeline

# function_code --------------------

def load_robotics_model():
    """
    Load the robotics model from Hugging Face.

    Returns:
        A pipeline object which is the loaded model.

    Raises:
        OSError: If the model 'Antheia/Hanna' does not exist or the file 'config.json' is not found.
    """
    try:
        robotics_pipeline = pipeline('robotics', model='Antheia/Hanna')
        return robotics_pipeline
    except OSError as e:
        print(f'Error: {e}')

# test_function_code --------------------

def test_load_robotics_model():
    """
    Test the function load_robotics_model.

    Raises:
        AssertionError: If the function does not return a pipeline object.
    """
    try:
        result = load_robotics_model()
        assert isinstance(result, type(pipeline('sentiment-analysis'))), 'The function does not return a pipeline object.'
        print('Test passed.')
    except AssertionError as e:
        print(f'Test failed: {e}')

# call_test_function_code --------------------

test_load_robotics_model()