# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def fill_mask_dutch_sentence(input_text: str) -> str:
    """
    Fill in the missing word in a Dutch sentence using a pre-trained BERT model.

    Args:
        input_text (str): The input sentence with a missing word represented as '___'.

    Returns:
        str: The complete sentence with the missing word filled in.

    Raises:
        OSError: If there is an issue with loading the pre-trained model or tokenizing the input.
    """
    tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
    model = AutoModel.from_pretrained('GroNLP/bert-base-dutch-cased')
    input_tokens = tokenizer.encode(input_text, return_tensors='pt')
    mask_position = input_tokens.tolist()[0].index(tokenizer.mask_token_id)
    output = model(input_tokens)
    prediction = output.logits.argmax(dim=2)[0].item()
    predicted_word = tokenizer.convert_ids_to_tokens(prediction)
    filled_sentence = input_text.replace('___', predicted_word)
    return filled_sentence

# test_function_code --------------------

def test_fill_mask_dutch_sentence():
    """
    Test the fill_mask_dutch_sentence function with various test cases.
    """
    assert fill_mask_dutch_sentence('Het is vandaag erg koud, dus vergeet niet je ___ mee te nemen.') != ''
    assert fill_mask_dutch_sentence('Ik ga naar de ___ om boodschappen te doen.') != ''
    assert fill_mask_dutch_sentence('Het weer is erg ___, dus neem een paraplu mee.') != ''
    return 'All Tests Passed'

# call_test_function_code --------------------

test_fill_mask_dutch_sentence()