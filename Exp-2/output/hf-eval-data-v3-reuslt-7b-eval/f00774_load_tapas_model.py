# function_import --------------------

from transformers import TapasForQuestionAnswering

# function_code --------------------

def load_tapas_model():
    """
    Load the pre-trained TAPAS model for table question answering.

    Returns:
        model: A pre-trained TAPAS model.
    """
    return TapasForQuestionAnswering.from_pretrained("google/tapas-base")

# test_function_code --------------------

def test_load_tapas_model():
    """
    Test the load_tapas_model function.
    """
    model = load_tapas_model()
    assert isinstance(model, TapasForQuestionAnswering), 'Model loading failed.'
    print('All Tests Passed')


# call_test_function_code --------------------

if __name__ == '__main__':
    test_load_tapas_model()