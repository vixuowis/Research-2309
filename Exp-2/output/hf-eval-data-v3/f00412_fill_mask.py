# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_mask(sentence):
    """
    Fill in the blanks in a given French sentence using the 'camembert-base' model.

    Args:
        sentence (str): The sentence with a masked token ('<mask>') that needs to be filled.

    Returns:
        list: A list of dictionaries with the filled sentence and the score of the prediction.
    """
    camembert_fill_mask = pipeline('fill-mask', model='camembert-base', tokenizer='camembert-base')
    results = camembert_fill_mask(sentence)
    return results

# test_function_code --------------------

def test_fill_mask():
    """
    Test the fill_mask function with some test cases.
    """
    assert fill_mask('Le camembert est <mask> :)')[0]['sequence'] == 'Le camembert est délicieux :)', 'Test Case 1 Failed'
    assert fill_mask('Paris est la <mask> de la France.')[0]['sequence'] == 'Paris est la capitale de la France.', 'Test Case 2 Failed'
    assert fill_mask('La Tour Eiffel est <mask>.')[0]['sequence'] == 'La Tour Eiffel est magnifique.', 'Test Case 3 Failed'
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_fill_mask())