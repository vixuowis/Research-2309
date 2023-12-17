# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def initialize_decision_transformer(model_name):
    """
    Initialize the Decision Transformer model for hopping control.

    Args:
        model_name (str): The name of the pre-trained model.

    Returns:
        object: The loaded Decision Transformer model.
    """
    return AutoModel.from_pretrained(model_name)

# test_function_code --------------------

def test_initialize_decision_transformer():
    print("Testing initialize_decision_transformer function.")
    model_name = 'edbeeching/decision-transformer-gym-hopper-medium'
    model = initialize_decision_transformer(model_name)
    assert model is not None, f"Failed to initialize model with name {model_name}"
    print("Test passed: Successfully initialized the Decision Transformer model.")

test_initialize_decision_transformer()