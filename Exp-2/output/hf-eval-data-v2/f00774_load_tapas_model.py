# function_import --------------------

from transformers import TapasForQuestionAnswering

# function_code --------------------

def load_tapas_model():
    """
    Load the pre-trained TAPAS model for table question answering.

    This function loads the TAPAS model that has been fine-tuned on the WikiSQL dataset.
    The model is capable of answering questions based on structured tables.

    Returns:
        model: A TAPAS model instance.
    """
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wikisql-supervised')
    return model

# test_function_code --------------------

def test_load_tapas_model():
    """
    Test the load_tapas_model function.

    This function tests the load_tapas_model function by loading the model and checking its type.
    """
    model = load_tapas_model()
    assert isinstance(model, TapasForQuestionAnswering), 'Model loading failed.'

# call_test_function_code --------------------

test_load_tapas_model()