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
    """
    tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
    model = AutoModel.from_pretrained('GroNLP/bert-base-dutch-cased')
    input_tokens = tokenizer(input_sentence, return_tensors='pt')
    outputs = model(**input_tokens)
    predictions = outputs[0]
    predicted_index = torch.argmax(predictions[0]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    return input_sentence.replace('[MASK]', predicted_token)

# test_function_code --------------------

def test_fill_mask_dutch_sentence():
    """
    This function tests the fill_mask_dutch_sentence function.
    """
    test_sentence = 'Hij ging naar de [MASK] om boodschappen te doen.'
    expected_output = 'Hij ging naar de winkel om boodschappen te doen.'
    assert fill_mask_dutch_sentence(test_sentence) == expected_output

# call_test_function_code --------------------

test_fill_mask_dutch_sentence()