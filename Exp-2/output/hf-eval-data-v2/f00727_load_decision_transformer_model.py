# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_decision_transformer_model(model_name: str):
    """
    Load a pretrained Decision Transformer model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the pretrained model.

    Returns:
        A pretrained model of the specified name.
    """
    return AutoModel.from_pretrained(model_name)

# test_function_code --------------------

def test_load_decision_transformer_model():
    """
    Test the function load_decision_transformer_model.

    The function should return a model of the correct type.
    """
    model = load_decision_transformer_model('edbeeching/decision-transformer-gym-hopper-medium')
    assert isinstance(model, AutoModel), 'The returned object is not a Decision Transformer model.'

# call_test_function_code --------------------

test_load_decision_transformer_model()