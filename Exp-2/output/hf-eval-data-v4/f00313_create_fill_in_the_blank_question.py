# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline


# function_code --------------------

def create_fill_in_the_blank_question(sentence, keyword):
    """
    Replace the specified keyword in a sentence with the '[MASK]' token and
    utilize the 'distilbert-base-multilingual-cased' model to predict
    possible fill-in-the-blank options.

    Parameters:
    sentence (str): The sentence to be masked.
    keyword (str): The word in the sentence that needs to be replaced with '[MASK]'.

    Returns:
    list: A list of dictionaries with possible words that can fill the blank.
    """
    # Replace the keyword with [MASK]
    masked_sentence = sentence.replace(keyword, '[MASK]')

    # Initialize the pipeline
    unmasker = pipeline('fill-mask', model='distilbert-base-multilingual-cased')

    # Use the unmasker pipeline to predict the masked word
    possible_words = unmasker(masked_sentence)

    return possible_words


# test_function_code --------------------

def test_create_fill_in_the_blank_question():
    print("Testing started.")
    # Test case 1: Simple sentence
    sentence = "The capital of France is [MASK]."
    keyword = "Paris"
    result = create_fill_in_the_blank_question(sentence, keyword)
    assert type(result) is list, "Test case failed: The result should be a list."

    # Test case 2: Another language example sentence
    sentence = "El capital de España es [MASK]."
    keyword = "Madrid"
    result = create_fill_in_the_blank_question(sentence, keyword)
    assert type(result) is list, "Test case failed: The result should be a list."

    print("Testing finished.")

# Run the test function
test_create_fill_in_the_blank_question()
