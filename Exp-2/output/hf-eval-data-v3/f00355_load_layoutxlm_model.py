# function_import --------------------

from transformers import LayoutXLMForQuestionAnswering

# function_code --------------------

def load_layoutxlm_model(model_name: str):
    """
    Load a pre-trained LayoutXLM model for Document Question Answering.

    Args:
        model_name (str): The name of the pre-trained model.

    Returns:
        LayoutXLMForQuestionAnswering: The loaded model.

    Raises:
        ImportError: If the transformers library is not installed or the model cannot be loaded.
    """
    try:
        model = LayoutXLMForQuestionAnswering.from_pretrained(model_name)
        return model
    except ImportError as e:
        print(f'Error: {e}')
        raise

# test_function_code --------------------

def test_load_layoutxlm_model():
    """
    Test the load_layoutxlm_model function.
    """
    model_name = 'fimu-docproc-research/CZ_DVQA_layoutxlm-base'
    try:
        model = load_layoutxlm_model(model_name)
        assert isinstance(model, LayoutXLMForQuestionAnswering), 'Model loading failed'
        print('Test passed')
    except Exception as e:
        print(f'Test failed: {e}')

# call_test_function_code --------------------

test_load_layoutxlm_model()