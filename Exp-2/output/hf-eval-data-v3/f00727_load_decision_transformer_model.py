# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_decision_transformer_model(model_name: str):
    """
    Load the pretrained Decision Transformer model.

    Args:
        model_name (str): The name of the pretrained model.

    Returns:
        A pretrained model of the specified name.

    Raises:
        OSError: If there is a problem with the disk space while loading the model.
    """
    try:
        return AutoModel.from_pretrained(model_name)
    except OSError as e:
        print(f'Error while loading the model: {e}')

# test_function_code --------------------

def test_load_decision_transformer_model():
    """
    Test the function load_decision_transformer_model.
    """
    model_name = 'edbeeching/decision-transformer-gym-hopper-medium'
    model = load_decision_transformer_model(model_name)
    assert model is not None, 'Model loading failed'
    print('All Tests Passed')

# call_test_function_code --------------------

test_load_decision_transformer_model()