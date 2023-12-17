# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def unmask_text(masked_text):
    # This function uses the Hugging Face transformers pipeline to predict
    # the masked words in a given text snippet using fill-mask model.

    # Initialize the pipeline with the xlm-roberta-large model
    unmasker = pipeline('fill-mask', model='xlm-roberta-large')

    # Use the unmasker pipeline to predict the masked words
    filled_result = unmasker(masked_text)

    # Return the result
    return filled_result

# test_function_code --------------------

def test_unmask_text():
    print("Testing started.")

    # Test case 1: Simple sentence with one mask
    masked_text_1 = "<mask> are large, slow-moving reptiles."
    expected_word_1 = "Alligators"  # Assuming the expected word for this context
    result_1 = unmask_text(masked_text_1)

    assert any(expected_word_1 in prediction["sequence"] for prediction in result_1), f"Test case 1 failed: {result_1}"

    # Test case 2: Sentence with two masks
    masked_text_2 = "They are well-adapted to life in <mask>, and they are a common sight."
    expected_word_2 = "water"  # Assuming the expected word for this context
    result_2 = unmask_text(masked_text_2)

    assert any(expected_word_2 in prediction["sequence"] for prediction in result_2), f"Test case 2 failed: {result_2}"

    print("Testing finished.")

# Run the test function
test_unmask_text()