# requirements_file --------------------

import subprocess

requirements = ["torch", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from parrot import Parrot
import torch

# function_code --------------------

def generate_paraphrases(phrase):
    """Generate paraphrased variations of the given phrase using the Parrot model.

    Args:
        phrase (str): The phrase to be paraphrased.

    Returns:
        list: A list of paraphrased sentences.

    Raises:
        Exception: If an error occurs in the API call or processing the paraphrases.

    """
    try:
        # Initializing the Parrot model
        parrot = Parrot(model_tag='prithivida/parrot_paraphraser_on_T5', use_gpu=False)

        # Generating paraphrases
        paraphrases = parrot.augment(input_phrase=phrase)

        # Extracting the paraphrased phrases
        paraphrased_phrases = [para_phrase[0] for para_phrase in paraphrases]
        return paraphrased_phrases
    except Exception as e:
        raise Exception(f'Error generating paraphrases: {str(e)}')

# test_function_code --------------------

def test_generate_paraphrases():
    print("Testing started.")
    test_phrase = "How can I improve my time management skills?"

    # Test case 1: Check if the function returns a list
    print("Testing case [1/3] started.")
    paraphrases = generate_paraphrases(test_phrase)
    assert isinstance(paraphrases, list), f"Test case [1/3] failed: Expected a list, got {type(paraphrases).__name__}."

    # Test case 2: Check if at least one paraphrase is generated
    print("Testing case [2/3] started.")
    assert len(paraphrases) > 0, "Test case [2/3] failed: No paraphrases generated."

    # Test case 3: Check if an exception is raised for an empty input
    print("Testing case [3/3] started.")
    try:
        generate_paraphrases("")
        assert False, "Test case [3/3] failed: Exception not raised for empty input."
    except Exception as e:
        assert "Error generating paraphrases" in str(e), f"Test case [3/3] failed: {str(e)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_paraphrases()