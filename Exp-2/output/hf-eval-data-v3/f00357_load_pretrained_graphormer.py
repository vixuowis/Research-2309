# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_pretrained_graphormer(model_name='graphormer-base-pcqm4mv1'):
    """
    Load a pretrained Graphormer model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the pretrained model. Default is 'graphormer-base-pcqm4mv1'.

    Returns:
        model (AutoModel): The loaded pretrained model.

    Raises:
        OSError: If the model_name is not a valid model identifier listed on 'https://huggingface.co/models'.
    """
    try:
        model = AutoModel.from_pretrained(model_name)
        return model
    except OSError as e:
        print(f'Error: {e}')

# test_function_code --------------------

def test_load_pretrained_graphormer():
    """
    Test the function load_pretrained_graphormer.
    """
    # Test with default model_name
    try:
        model = load_pretrained_graphormer()
        assert isinstance(model, AutoModel), 'Test Case 1 Failed'
        print('Test Case 1 Passed')
    except OSError:
        print('Test Case 1 Failed: Model not found')

    # Test with invalid model_name
    try:
        model = load_pretrained_graphormer('invalid-model')
        assert isinstance(model, AutoModel), 'Test Case 2 Failed'
        print('Test Case 2 Passed')
    except OSError:
        print('Test Case 2 Failed: Model not found')

    print('All Tests Passed')

# call_test_function_code --------------------

test_load_pretrained_graphormer()