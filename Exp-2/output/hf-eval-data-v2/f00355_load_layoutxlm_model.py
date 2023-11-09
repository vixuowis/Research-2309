# function_import --------------------

from transformers import LayoutXLMForQuestionAnswering

# function_code --------------------

def load_layoutxlm_model(model_name: str):
    """
    Load a pre-trained LayoutXLM model for Document Question Answering.

    Args:
        model_name (str): The name of the pre-trained model. 

    Returns:
        A pre-trained LayoutXLM model.
    """
    model = LayoutXLMForQuestionAnswering.from_pretrained(model_name)
    return model

# test_function_code --------------------

def test_load_layoutxlm_model():
    """
    Test the function load_layoutxlm_model.
    """
    model_name = 'fimu-docproc-research/CZ_DVQA_layoutxlm-base'
    model = load_layoutxlm_model(model_name)
    assert isinstance(model, LayoutXLMForQuestionAnswering), 'Model loading failed.'

# call_test_function_code --------------------

test_load_layoutxlm_model()