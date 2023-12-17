# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def assess_paraphrase_adequacy(generated_paraphrase):
    """
    Assess the adequacy of a generated paraphrase using a pretrained model.

    Args:
        generated_paraphrase (str): The paraphrase text to be assessed.

    Returns:
        dict: The classification result containing labels and scores.

    Raises:
        ValueError: If the generated_paraphrase is not a string.
    """
    if not isinstance(generated_paraphrase, str):
        raise ValueError('The paraphrase must be a string.')
    adequacy_classifier = pipeline('text-classification', model='prithivida/parrot_adequacy_model')
    return adequacy_classifier(generated_paraphrase)

# test_function_code --------------------

def test_assess_paraphrase_adequacy():
    print('Testing started.')

    # Test case 1: Valid paraphrase text
    print('Testing case [1/2] started.')
    paraphrase = 'This is a sample paraphrase.'
    result = assess_paraphrase_adequacy(paraphrase)
    assert isinstance(result, dict) and 'label' in result and 'score' in result, f'Test case [1/2] failed: Invalid result format {result}'

    # Test case 2: Invalid paraphrase (not a string)
    print('Testing case [2/2] started.')
    non_string_input = 12345
    try:
        assess_paraphrase_adequacy(non_string_input)
        assert False, 'Test case [2/2] failed: ValueError not raised for non-string input.'
    except ValueError as e:
        assert str(e) == 'The paraphrase must be a string.', f'Test case [2/2] failed: Unexpected ValueError message {e}'
    print('Testing finished.')

# call_test_function_line --------------------

test_assess_paraphrase_adequacy()