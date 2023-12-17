# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_synonyms(word):
    """
    Generate synonyms for a given word using a fill-mask language model.

    Parameters:
        word (str): The word for which to generate synonyms.

    Returns:
        list: A list of synonyms.
    """
    fill_mask = pipeline('fill-mask', model='microsoft/deberta-base')
    masked_sentence = f'He was feeling [MASK].'
    results = fill_mask(masked_sentence.replace('happy', word))
    synonyms = [result['sequence'].replace('He was feeling ', '').replace('.', '').strip() for result in results]
    return synonyms

# test_function_code --------------------

def test_generate_synonyms():
    print("Testing started.")
    word = 'happy'
    synonyms = generate_synonyms(word)

    # Testing case 1: Check if the returned value is a list
    print("Testing case [1/1] started.")
    assert isinstance(synonyms, list), f"Test case [1/1] failed: Expected a list, but got {type(synonyms).__name__}."
    print("Testing finished.")

# Run the test function
test_generate_synonyms()