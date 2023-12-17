# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def complete_french_sentence(sentence_with_mask):
    # Load the 'camembert-base' model and tokenizer for fill-mask task
    camembert_fill_mask = pipeline('fill-mask', model='camembert-base', tokenizer='camembert-base')

    # Use the model to fill in the missing word in the sentence
    results = camembert_fill_mask(sentence_with_mask)

    # The results are returned as a list of dictionaries; return the top result
    return results[0]['sequence']


# test_function_code --------------------

def test_complete_french_sentence():
    print("Testing started.")

    # Test case 1: Sentence with one mask
    print("Testing case [1/3] started.")
    result = complete_french_sentence("Le camembert est <mask>. :)")
    assert '<mask>' not in result, f"Test case [1/3] failed: The missing word was not filled. Result: {result}"

    # Test case 2: Sentence with no mask - should return the sentence unchanged
    print("Testing case [2/3] started.")
    sentence = "Il fait beau aujourd'hui."
    result = complete_french_sentence(sentence)
    assert result == sentence, f"Test case [2/3] failed: The sentence was changed. Result: {result}"

    # Test case 3: Sentence with multiple masks - should fill in all missing words
    print("Testing case [3/3] started.")
    result = complete_french_sentence("<mask> est la capital de la France et elle est <mask>.")
    assert '<mask>' not in result, f"Test case [3/3] failed: Not all missing words were filled. Result: {result}"
    print("Testing finished.")

# Run the test function
test_complete_french_sentence()
