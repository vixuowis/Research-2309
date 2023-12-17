# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def complete_the_sentence(sentence):
    """
    Complete the given English sentence by filling in the mask with the most probable word using a pre-trained model.

    Parameters:
        sentence (str): The input sentence with a <mask> token.

    Returns:
        str: The sentence with the <mask> replaced by the predicted word.
    """
    unmasker = pipeline('fill-mask', model='roberta-base')
    results = unmasker(sentence)
    # The results contain a list of possible words with scores; we'll pick the top one.
    completed_sentence = sentence.replace('<mask>', results[0]['token_str'])
    return completed_sentence

# test_function_code --------------------

def test_complete_the_sentence():
    print("Testing started.")
    sentence = "In the story, the protagonist's actions were <mask> beyond measure."

    # Testing case
    print("Testing case [1/1] started.")
    completed = complete_the_sentence(sentence)
    assert '<mask>' not in completed, f"Test case failed: Mask not filled."
    print("Testing finished.")