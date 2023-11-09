# function_import --------------------

from parrot import Parrot
import torch

# function_code --------------------

def generate_paraphrases(phrase):
    """
    This function generates paraphrases of a given phrase using the Parrot paraphraser.

    Args:
        phrase (str): The phrase to be paraphrased.

    Returns:
        list: A list of paraphrased sentences.
    """
    parrot = Parrot(model_tag='prithivida/parrot_paraphraser_on_T5', use_gpu=False)
    para_phrases = parrot.augment(input_phrase=phrase)
    return para_phrases

# test_function_code --------------------

def test_generate_paraphrases():
    """
    This function tests the generate_paraphrases function.
    It uses a sample phrase and checks if the output is a list.
    """
    phrase = 'How can I improve my time management skills?'
    paraphrases = generate_paraphrases(phrase)
    assert isinstance(paraphrases, list), 'The output should be a list.'

# call_test_function_code --------------------

test_generate_paraphrases()