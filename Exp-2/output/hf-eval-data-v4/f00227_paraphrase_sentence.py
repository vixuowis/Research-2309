# requirements_file --------------------

!pip install -U torch transformers

# function_import --------------------

from parrot import Parrot
import torch

# function_code --------------------

def paraphrase_sentence(input_phrase):
    """
    Generate paraphrases for a given input phrase using Parrot model.

    Parameters:
    input_phrase (str): The sentence to be paraphrased.

    Returns:
    list: A list of paraphrased sentences.
    """
    # Initialize the Parrot instance
    parrot = Parrot(model_tag='prithivida/parrot_paraphraser_on_T5', use_gpu=False)
    # Generate paraphrases
    paraphrases = parrot.augment(input_phrase=input_phrase)
    # Return the list of generated paraphrases
    return paraphrases

# test_function_code --------------------

def test_paraphrase_sentence():
    print("Testing paraphrase_sentence function.")
    test_phrase = "How can I improve my time management skills?"
    paraphrases = paraphrase_sentence(test_phrase)
    # Test case: The function should return a list.
    assert isinstance(paraphrases, list), "The function should return a list."
    # Test case: The list should not be empty.
    assert len(paraphrases) > 0, "The function should return at least one paraphrase."
    print("All tests passed!")

# Run the test function
test_paraphrase_sentence()