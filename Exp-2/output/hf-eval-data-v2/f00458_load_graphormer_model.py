# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_graphormer_model(model_name: str):
    """
    Load a pre-trained Graphormer model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the pre-trained model. For example, 'clefourrier/graphormer-base-pcqm4mv2'.

    Returns:
        A pre-trained model of the specified name.
    """
    model = AutoModel.from_pretrained(model_name)
    return model

# test_function_code --------------------

def test_load_graphormer_model():
    """
    Test the function load_graphormer_model.

    This function does not return anything but raises an error if the function load_graphormer_model is incorrect.
    """
    model_name = 'clefourrier/graphormer-base-pcqm4mv2'
    model = load_graphormer_model(model_name)
    assert model is not None, 'The model should not be None.'
    assert isinstance(model, AutoModel), 'The model should be an instance of AutoModel.'

# call_test_function_code --------------------

test_load_graphormer_model()