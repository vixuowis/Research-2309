# function_import --------------------

from parrot import Parrot
import torch

# function_code --------------------

def generate_paraphrases(phrase: str, model_tag: str = 'prithivida/parrot_paraphraser_on_T5', use_gpu: bool = False):
    """
    Generate paraphrases for a given phrase using the Parrot paraphraser.

    Args:
        phrase (str): The phrase to paraphrase.
        model_tag (str, optional): The model tag of the Parrot paraphraser. Defaults to 'prithivida/parrot_paraphraser_on_T5'.
        use_gpu (bool, optional): Whether to use GPU for paraphrasing. Defaults to False.

    Returns:
        list: A list of paraphrased phrases.
    """
    parrot = Parrot(model_tag=model_tag, use_gpu=use_gpu)
    para_phrases = parrot.augment(input_phrase=phrase)
    return para_phrases

# test_function_code --------------------

def test_generate_paraphrases():
    """
    Test the generate_paraphrases function.
    """
    phrase = 'How can I improve my time management skills?'
    paraphrases = generate_paraphrases(phrase)
    assert isinstance(paraphrases, list), 'The return type should be a list.'
    assert len(paraphrases) > 0, 'The list of paraphrases should not be empty.'
    for paraphrase in paraphrases:
        assert isinstance(paraphrase, str), 'Each paraphrase should be a string.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_paraphrases()