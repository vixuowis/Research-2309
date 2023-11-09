# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_pretrained_graphormer(model_name: str = 'graphormer-base-pcqm4mv1'):
    """
    Load a pretrained Graphormer model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the pretrained model. Default is 'graphormer-base-pcqm4mv1'.

    Returns:
        A Graphormer model instance.
    """
    model = AutoModel.from_pretrained(model_name)
    return model

# test_function_code --------------------

def test_load_pretrained_graphormer():
    """
    Test the function load_pretrained_graphormer.

    This function does not return anything but raises an error if the loaded model is not an instance of transformers.PreTrainedModel.
    """
    model = load_pretrained_graphormer()
    assert isinstance(model, transformers.PreTrainedModel), 'The loaded model is not a Graphormer model.'

# call_test_function_code --------------------

test_load_pretrained_graphormer()