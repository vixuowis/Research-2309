# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_graphormer_model(model_name='graphormer-base-pcqm4mv1'):
    """
    Load the Graphormer model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the model to load. Default is 'graphormer-base-pcqm4mv1'.

    Returns:
        A Graphormer model.

    Raises:
        OSError: If the model_name is not a valid model identifier listed on 'https://huggingface.co/models'
    """

    return AutoModel.from_pretrained(model_name)


# test_function_code --------------------

def test_load_graphormer_model():
    """
    Test the load_graphormer_model function.
    """
    try:
        # Test with default model_name
        model = load_graphormer_model()
        assert model is not None, 'Model should not be None'

        # Test with a non-existent model_name
        model = load_graphormer_model('non-existent-model')
        assert model is None, 'Model should be None for non-existent model'

        print('All Tests Passed')
    except Exception as e:
        print(f'Test Failed: {e}')


# call_test_function_code --------------------

test_load_graphormer_model()