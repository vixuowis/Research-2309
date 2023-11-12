# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_graphormer_model(model_name: str):
    """
    Load a pre-trained Graphormer model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the pre-trained model.

    Returns:
        transformers.PreTrainedModel: The loaded model.

    Raises:
        ValueError: If the model name is not provided or the model cannot be loaded.
    """
    if not model_name:
        raise ValueError('No model name provided.')

    try:
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        raise ValueError(f'Failed to load model: {e}')

    return model

# test_function_code --------------------

def test_load_graphormer_model():
    """
    Test the load_graphormer_model function.
    """
    # Test with a valid model name
    model = load_graphormer_model('clefourrier/graphormer-base-pcqm4mv2')
    assert isinstance(model, AutoModel), 'The model should be an instance of AutoModel.'

    # Test with an invalid model name
    try:
        model = load_graphormer_model('invalid_model_name')
    except ValueError:
        pass
    else:
        assert False, 'Expected a ValueError when loading an invalid model name.'

    # Test with no model name
    try:
        model = load_graphormer_model('')
    except ValueError:
        pass
    else:
        assert False, 'Expected a ValueError when no model name is provided.'

    print('All tests passed.')

# call_test_function_code --------------------

test_load_graphormer_model()