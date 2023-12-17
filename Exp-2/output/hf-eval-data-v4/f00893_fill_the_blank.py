# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------


# Initialize the pipeline for fill-mask task using BERT large uncased model
fill_in_the_blanks = pipeline('fill-mask', model='bert-large-uncased')

def fill_the_blank(sentence):
    """
    This function takes a sentence with a '[MASK]' token and uses the BERT model to predict 
    and fill in the missing word.

    Args:
    - sentence (str): A sentence with a '[MASK]' token representing the missing word.

    Returns:
    - filled_sentence (str): The sentence with the '[MASK]' token replaced by the predicted word.
    """
    # Use the model to fill the blank
    filled_sentence = fill_in_the_blanks(sentence)

    return filled_sentence

# test_function_code --------------------


def test_fill_in_the_blanks():
    print("Testing started.")
    
    # Test case 1: Single mask in a sentence
    print("Testing case [1/3] started.")
    sentence_1 = "The quick brown fox jumps over the lazy [MASK]."
    result_1 = fill_the_blank(sentence_1)
    assert result_1[0]['sequence'] == "the quick brown fox jumps over the lazy dog.", \
        f"Test case [1/3] failed: expected 'dog' but got {result_1[0]['sequence']}"
    print("Test case [1/3] passed.")

    # Test case 2: Proper noun mask
    print("Testing case [2/3] started.")
    sentence_2 = "[MASK] was the first president of the United States."
    result_2 = fill_the_blank(sentence_2)
    assert result_2[0]['sequence'].lower().strip() == "washington was the first president of the united states.", \
        f"Test case [2/3] failed: expected 'Washington' but got {result_2[0]['sequence']}"
    print("Test case [2/3] passed.")

    # Test case 3: Common noun mask
    print("Testing case [3/3] started.")
    sentence_3 = "I need to buy a new [MASK] for my computer."
    result_3 = fill_the_blank(sentence_3)
    assert result_3[0]['sequence'] == "i need to buy a new computer for my computer.", \
        f"Test case [3/3] failed: expected 'computer' but got {result_3[0]['sequence']}"
    print("Test case [3/3] passed.")
    
    print("Testing finished.")

# Run the test function
test_fill_in_the_blanks()