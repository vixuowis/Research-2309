# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_depth_estimation_model(model_name: str):
    """
    Load a pretrained depth estimation model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the pretrained model. For example, 'sayakpaul/glpn-nyu-finetuned-diode-221122-082237'.

    Returns:
        A pretrained model for depth estimation.
    """
    depth_estimation_model = AutoModel.from_pretrained(model_name)
    return depth_estimation_model

# test_function_code --------------------

def test_load_depth_estimation_model():
    """
    Test the function 'load_depth_estimation_model'.
    """
    model_name = 'sayakpaul/glpn-nyu-finetuned-diode-221122-082237'
    model = load_depth_estimation_model(model_name)
    assert isinstance(model, AutoModel), 'The returned object is not an instance of AutoModel.'

# call_test_function_code --------------------

test_load_depth_estimation_model()