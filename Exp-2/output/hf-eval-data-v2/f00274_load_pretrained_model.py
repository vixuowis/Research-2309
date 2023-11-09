# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_pretrained_model(model_name: str):
    """
    Load a pretrained model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the pretrained model. For example, 'sayakpaul/glpn-nyu-finetuned-diode-221122-082237'.

    Returns:
        A pretrained model.
    """
    model = AutoModel.from_pretrained(model_name)
    return model

# test_function_code --------------------

def test_load_pretrained_model():
    """
    Test the function load_pretrained_model.

    This function does not return anything but raises an error if the function load_pretrained_model is incorrect.
    """
    model_name = 'sayakpaul/glpn-nyu-finetuned-diode-221122-082237'
    model = load_pretrained_model(model_name)
    assert model is not None, 'The model should not be None.'
    assert isinstance(model, AutoModel), 'The model should be an instance of AutoModel.'

# call_test_function_code --------------------

test_load_pretrained_model()