# function_import --------------------

from transformers import pipeline

# function_code --------------------

def assess_paraphrase_adequacy(generated_paraphrase):
    """
    This function uses a pretrained model to assess the adequacy of a paraphrased text.

    Args:
        generated_paraphrase (str): The paraphrased text to be assessed.

    Returns:
        dict: The classification result from the model. The keys are the class labels, and the values are the corresponding scores.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(generated_paraphrase, str):
        raise ValueError('Input to assess_paraphrase_adequacy must be a string.')
    adequacy_classifier = pipeline('text-classification', model='prithivida/parrot_adequacy_model')
    return adequacy_classifier(generated_paraphrase)

# test_function_code --------------------

def test_assess_paraphrase_adequacy():
    """
    This function tests the assess_paraphrase_adequacy function.
    It uses a predefined paraphrase and checks if the output is a dictionary.
    """
    test_paraphrase = 'This is a test paraphrase.'
    result = assess_paraphrase_adequacy(test_paraphrase)
    assert isinstance(result, dict), 'The output of assess_paraphrase_adequacy should be a dictionary.'

# call_test_function_code --------------------

test_assess_paraphrase_adequacy()