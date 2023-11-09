# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_graphormer_model(model_name: str = 'graphormer-base-pcqm4mv1'):
    """
    Load the pretrained Graphormer model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the pretrained model. Default is 'graphormer-base-pcqm4mv1'.

    Returns:
        A pretrained Graphormer model.
    """
    model = AutoModel.from_pretrained(model_name)
    return model

# test_function_code --------------------

def test_load_graphormer_model():
    """
    Test the function load_graphormer_model.
    """
    model = load_graphormer_model()
    assert model is not None, 'Model loading failed.'
    assert isinstance(model, AutoModel), 'The loaded model is not an instance of AutoModel.'

# call_test_function_code --------------------

test_load_graphormer_model()