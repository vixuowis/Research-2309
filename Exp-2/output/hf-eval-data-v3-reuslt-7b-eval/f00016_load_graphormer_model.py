# function_import --------------------

from transformers import AutoModel

# function_code --------------------

def load_graphormer_model(model_name='graphormer-base-pcqm4mv1'):
    """
    Load the Graphormer model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the model to load. Default is 'graphormer-base-pcqm4mv1'.

    Returns:
        A Graphormer model.

    Raises:
        OSError: If the model_name is not a valid model identifier listed on 'https://huggingface.co/models'
    """
    
    # Check if the name is available on Hugging Face Transformers. 
    model_list = ['graphormer-sparse', 'graphormer-base']
    for i in range(len(model_list)):
        model_list[i] += '-pcqm4mv1'
        
    if not (model_name in model_list):
        raise OSError('Invalid model name. Available models: ' + str(model_list))
    
    # Load the Graphormer model from Hugging Face Transformers.
    model = AutoModel.from_pretrained(model_name)
    
    return model

# test_function_code --------------------

def test_load_graphormer_model():
    """
    Test the load_graphormer_model function.
    """
    try:
        # Test with default model_name
        model = load_graphormer_model()
        assert model is not None, 'Model should not be None'

        # Test with a non-existent model_name
        model = load_graphormer_model('non-existent-model')
        assert model is None, 'Model should be None for non-existent model'

        print('All Tests Passed')
    except Exception as e:
        print(f'Test Failed: {e}')


# call_test_function_code --------------------

test_load_graphormer_model()