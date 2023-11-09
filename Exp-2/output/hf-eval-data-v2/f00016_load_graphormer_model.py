# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_graphormer_model(model_name: str = 'graphormer-base-pcqm4mv1'):
    """
    Load the Graphormer model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the model to load. Default is 'graphormer-base-pcqm4mv1'.

    Returns:
        A Graphormer model instance.
    """
    return AutoModel.from_pretrained(model_name)

# test_function_code --------------------

def test_load_graphormer_model():
    """
    Test the load_graphormer_model function.
    """
    model = load_graphormer_model()
    assert model is not None, 'Model loading failed.'
    assert isinstance(model, AutoModel), 'Loaded model is not an instance of AutoModel.'

# call_test_function_code --------------------

test_load_graphormer_model()