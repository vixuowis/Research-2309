# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def fill_mask_with_legal_bert(text):
    """
    This function uses the 'nlpaueb/legal-bert-small-uncased' model to fill in the gaps in a legal document.
    The model is a lightweight version of the BERT-BASE model, providing higher efficiency while maintaining a high level of accuracy.

    Args:
        text (str): The legal document with gaps to be filled.

    Returns:
        str: The legal document with the gaps filled.
    """
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-small-uncased')
    model = AutoModel.from_pretrained('nlpaueb/legal-bert-small-uncased')
    inputs = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True)
    prediction = model(**inputs)
    predicted_index = torch.argmax(prediction[0], axis=-1)
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    return text.replace('[MASK]', predicted_token)

# test_function_code --------------------

def test_fill_mask_with_legal_bert():
    """
    This function tests the 'fill_mask_with_legal_bert' function with a sample legal document.
    """
    test_text = 'The defendant is [MASK] guilty of the crime.'
    expected_output = 'The defendant is not guilty of the crime.'
    assert fill_mask_with_legal_bert(test_text) == expected_output

# call_test_function_code --------------------

test_fill_mask_with_legal_bert()