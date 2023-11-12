# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_decision_transformer_model(model_name: str):
    """
    Load the Decision Transformer model from the Hugging Face model hub.

    Args:
        model_name (str): The name of the pretrained model.

    Returns:
        transformers.PreTrainedModel: The loaded model.
    """
    model = AutoModel.from_pretrained(model_name)
    return model

# test_function_code --------------------

def test_load_decision_transformer_model():
    """
    Test the load_decision_transformer_model function.
    """
    model_name = 'edbeeching/decision-transformer-gym-walker2d-expert'
    model = load_decision_transformer_model(model_name)
    assert isinstance(model, AutoModel), 'Loaded model is not of the correct type.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_load_decision_transformer_model()