# function_import --------------------

from transformers import Pix2StructForConditionalGeneration, T5Tokenizer, T5Config

# function_code --------------------

def load_pix2struct_model(model_name='google/pix2struct-chartqa-base'):
    """
    Load the Pix2Struct model for visual question answering.

    Args:
        model_name (str): The name of the pre-trained model. Default is 'google/pix2struct-chartqa-base'.

    Returns:
        model (Pix2StructForConditionalGeneration): The loaded Pix2Struct model.
        tokenizer (T5Tokenizer): The tokenizer associated with the model.

    Raises:
        ValueError: If the model cannot be loaded.
    """
    try:
        config = T5Config.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = Pix2StructForConditionalGeneration.from_pretrained(model_name, config=config)
        return model, tokenizer
    except Exception as e:
        raise ValueError(f'Unable to load model: {e}')

# test_function_code --------------------

def test_load_pix2struct_model():
    """
    Test the load_pix2struct_model function.
    """
    try:
        model, tokenizer = load_pix2struct_model()
        assert isinstance(model, Pix2StructForConditionalGeneration), 'Model loading failed'
        assert isinstance(tokenizer, T5Tokenizer), 'Tokenizer loading failed'
        print('All Tests Passed')
    except Exception as e:
        print(f'Test failed: {e}')

# call_test_function_code --------------------

test_load_pix2struct_model()