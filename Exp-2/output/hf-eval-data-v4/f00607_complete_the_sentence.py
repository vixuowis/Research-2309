# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def complete_the_sentence(sentence_part):
    """
    Complete a given sentence starting with "Hello, I'm a ..." using BERT model.

    Parameters:
    sentence_part (str): The initial part of the sentence to be completed.

    Returns:
    str: The completed sentence.
    """
    # Load the fill-mask pipeline with the bert-large-cased model
    unmasker = pipeline('fill-mask', model='bert-large-cased')

    # Append the mask token to the input text
    input_text = sentence_part + ' [MASK]...'

    # Pass the input text to the unmasker pipeline to get the completed sentence
    completed_sentence = unmasker(input_text)

    # Return only the top result's sequence
    return completed_sentence[0]['sequence']

# test_function_code --------------------

def test_complete_the_sentence():
    print("Testing started.")

    # Test case 1: Check for coherent sentence completion
    input_text = "Hello, I'm a"
    expected_output_contains = "Hello, I'm a"
    completed_sentence = complete_the_sentence(input_text)

    print("Testing case [1/1] started.")
    assert expected_output_contains in completed_sentence, f"Test case [1/1] failed: Expected output to contain '{expected_output_contains}', but got '{completed_sentence}'."

    print("Testing finished.")

# Run the test function
test_complete_the_sentence()