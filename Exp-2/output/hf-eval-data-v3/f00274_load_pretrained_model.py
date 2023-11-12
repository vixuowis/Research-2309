# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_pretrained_model(model_name: str):
    """
    Load a pretrained model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the pretrained model.

    Returns:
        An instance of the pretrained model.
    """
    model = AutoModel.from_pretrained(model_name)
    return model

# test_function_code --------------------

def test_load_pretrained_model():
    """
    Test the load_pretrained_model function.
    """
    model = load_pretrained_model('sayakpaul/glpn-nyu-finetuned-diode-221122-082237')
    assert isinstance(model, AutoModel), 'The model should be an instance of AutoModel.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_load_pretrained_model()