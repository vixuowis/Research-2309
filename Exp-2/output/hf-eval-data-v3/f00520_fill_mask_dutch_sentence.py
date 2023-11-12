# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def fill_mask_dutch_sentence(input_sentence: str) -> str:
    """
    This function takes a Dutch sentence with a masked token and returns the sentence with the masked token replaced by the most suitable word.

    Args:
        input_sentence (str): A Dutch sentence with a masked token represented by [MASK].

    Returns:
        str: The input sentence with the masked token replaced by the most suitable word.

    Raises:
        OSError: If there is not enough disk space to download the model.
    """
    tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
    model = AutoModel.from_pretrained('GroNLP/bert-base-dutch-cased')
    input_tokens = tokenizer(input_sentence, return_tensors='pt')
    outputs = model(**input_tokens)
    return outputs

# test_function_code --------------------

def test_fill_mask_dutch_sentence():
    """
    This function tests the fill_mask_dutch_sentence function with some test cases.
    """
    test_sentence_1 = 'Hij ging naar de [MASK] om boodschappen te doen.'
    expected_output_1 = 'Hij ging naar de winkel om boodschappen te doen.'
    assert fill_mask_dutch_sentence(test_sentence_1) == expected_output_1

    test_sentence_2 = 'Het weer is vandaag erg [MASK].'
    expected_output_2 = 'Het weer is vandaag erg mooi.'
    assert fill_mask_dutch_sentence(test_sentence_2) == expected_output_2

    test_sentence_3 = 'Ik ga naar de [MASK] om een boek te lenen.'
    expected_output_3 = 'Ik ga naar de bibliotheek om een boek te lenen.'
    assert fill_mask_dutch_sentence(test_sentence_3) == expected_output_3

    return 'All Tests Passed'

# call_test_function_code --------------------

test_fill_mask_dutch_sentence()