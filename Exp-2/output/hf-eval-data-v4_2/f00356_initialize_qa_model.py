# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForQuestionAnswering

# function_code --------------------

def initialize_qa_model(pretrained_model_name):
    """
    Initializes the question answering model with the given pretrained model name.

    Args:
        pretrained_model_name (str): The name of the pretrained model to load.

    Returns:
        model: The loaded question answering model.

    Raises:
        ValueError: If the pretrained model name is empty or None.
    """
    if not pretrained_model_name:
        raise ValueError('The pretrained model name cannot be empty or None.')
    model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name)
    return model


# test_function_code --------------------

def test_initialize_qa_model():
    print('Testing started.')
    
    # Test case 1: Valid model name
    print('Testing case [1/2] started.')
    try:
        model = initialize_qa_model('hf-tiny-model-private/tiny-random-LayoutLMForQuestionAnswering')
        assert model is not None, 'Test case [1/2] failed: Model should not be None.'
    except Exception as e:
        assert False, f'Test case [1/2] failed: {str(e)}'
    
    # Test case 2: Invalid model name (None)
    print('Testing case [2/2] started.')
    try:
        initialize_qa_model(None)
        assert False, 'Test case [2/2] failed: Should raise ValueError.'
    except ValueError:
        pass
    except Exception as e:
        assert False, f'Test case [2/2] failed: Expected ValueError, got {type(e).__name__}.'
    print('Testing finished.')


# call_test_function_line --------------------

test_initialize_qa_model()