# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def complete_marketing_slogan(slogan_with_mask):
    """
    Complete a marketing slogan with a masked word using a pre-trained language model.
    
    Parameters:
        slogan_with_mask (str): The marketing slogan with a masked word, denoted by <mask>.
    
    Returns:
        str: The marketing slogan with the mask filled by the most probable word.
    """
    # Create a fill-mask pipeline using the 'roberta-large' model
    unmasker = pipeline('fill-mask', model='roberta-large')
    # Generate a list of suggestions to complete the slogan
    suggestions = unmasker(slogan_with_mask)
    # The unmasked slogan with the highest probability will be the suggested completion
    completed_slogan = suggestions[0]['sequence']
    return completed_slogan

# test_function_code --------------------

def test_complete_marketing_slogan():
    print("Testing started.")
    # Test case 1: Slogan with common placeholder
    print("Testing case [1/1] started.")
    slogan_test = "Customer satisfaction is our top <mask>."
    expected_slogan = "Customer satisfaction is our top priority."
    completed_slogan = complete_marketing_slogan(slogan_test)
    assert completed_slogan == expected_slogan, f"Test case [1/1] failed: Expected '{{expected_slogan}}', but got '{{completed_slogan}}'"
    print("Testing finished.")

# Run the test function
test_complete_marketing_slogan()