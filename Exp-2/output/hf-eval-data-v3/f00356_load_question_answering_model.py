# function_import --------------------

from transformers import AutoModelForQuestionAnswering

# function_code --------------------

def load_question_answering_model(model_name: str):
    """
    Load a question answering model from the transformers library.

    Args:
        model_name (str): The name of the model to load. This should be a string that specifies the name of a pre-trained model available in the transformers library.

    Returns:
        A transformers.AutoModelForQuestionAnswering instance.

    Raises:
        ValueError: If the model_name is not a string.
        OSError: If the model_name does not correspond to a pre-trained model available in the transformers library.
    """
    if not isinstance(model_name, str):
        raise ValueError('model_name must be a string')
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    except OSError as e:
        raise OSError('Could not load model. Please make sure the model name is correct and the model is available in the transformers library.') from e
    return model

# test_function_code --------------------

def test_load_question_answering_model():
    """
    Test the load_question_answering_model function.
    """
    # Test with a valid model name
    model = load_question_answering_model('bert-base-uncased')
    assert isinstance(model, AutoModelForQuestionAnswering), 'Model is not an instance of AutoModelForQuestionAnswering'

    # Test with an invalid model name
    try:
        model = load_question_answering_model('invalid-model-name')
    except OSError:
        pass
    else:
        assert False, 'Expected an OSError when loading an invalid model name'

    # Test with a non-string model name
    try:
        model = load_question_answering_model(123)
    except ValueError:
        pass
    else:
        assert False, 'Expected a ValueError when loading a model with a non-string name'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_load_question_answering_model()