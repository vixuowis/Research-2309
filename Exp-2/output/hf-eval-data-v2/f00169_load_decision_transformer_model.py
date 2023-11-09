# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_decision_transformer_model(model_name: str):
    """
    Load the Decision Transformer model from the Hugging Face model hub.

    Args:
        model_name (str): The name of the model to be loaded. For example, 'edbeeching/decision-transformer-gym-walker2d-expert'.

    Returns:
        model: The loaded model.
    """
    model = AutoModel.from_pretrained(model_name)
    return model

# test_function_code --------------------

def test_load_decision_transformer_model():
    """
    Test the function load_decision_transformer_model.
    """
    model_name = 'edbeeching/decision-transformer-gym-walker2d-expert'
    model = load_decision_transformer_model(model_name)
    assert model is not None, 'Model loading failed.'
    assert isinstance(model, AutoModel), 'Loaded model is not of the correct type.'

# call_test_function_code --------------------

test_load_decision_transformer_model()