# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_graphormer_model(model_name='graphormer-base-pcqm4mv1'):
    """
    Load the pretrained Graphormer model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the pretrained model. Default is 'graphormer-base-pcqm4mv1'.

    Returns:
        model (AutoModel): The loaded Graphormer model.

    Raises:
        OSError: If the model_name is not a valid model identifier listed on 'https://huggingface.co/models'
    """
    try:
        model = AutoModel.from_pretrained(model_name)
        return model
    except OSError as e:
        print(f'Error: {e}')

# test_function_code --------------------

def test_load_graphormer_model():
    """
    Test the function load_graphormer_model.
    """
    try:
        # Test with default model_name
        model = load_graphormer_model()
        assert model is not None, 'Model loading failed with default model_name'

        # Test with a valid model_name
        model = load_graphormer_model('bert-base-uncased')
        assert model is not None, 'Model loading failed with valid model_name'

        # Test with an invalid model_name
        model = load_graphormer_model('invalid-model-name')
        assert model is None, 'Model loading should fail with invalid model_name'

        print('All Tests Passed')
    except AssertionError as e:
        print(f'Test failed: {e}')

# call_test_function_code --------------------

if __name__ == '__main__':
    test_load_graphormer_model()