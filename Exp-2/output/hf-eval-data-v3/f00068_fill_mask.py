# function_import --------------------

from transformers import AutoTokenizer, AutoModelForMaskedLM

# function_code --------------------

def fill_mask(masked_text: str) -> str:
    """
    Fill in the missing word in a given Japanese text using a pretrained BERT model.

    Args:
        masked_text (str): The input text with a masked word represented as '[MASK]'.

    Returns:
        str: The input text with the masked word replaced by the predicted word.
    """
    tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    model = AutoModelForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese')
    encoded_input = tokenizer(masked_text, return_tensors='pt')
    outputs = model(**encoded_input)
    prediction = outputs.logits.argmax(-1)
    predicted_token = tokenizer.convert_ids_to_tokens(prediction[0])
    filled_text = masked_text.replace('[MASK]', predicted_token[1])
    return filled_text

# test_function_code --------------------

def test_fill_mask():
    """
    Test the fill_mask function with some test cases.
    """
    assert fill_mask('テキストに[MASK]語があります。') != 'テキストに[MASK]語があります。'
    assert fill_mask('私の名前は[MASK]です。') != '私の名前は[MASK]です。'
    assert fill_mask('今日は[MASK]です。') != '今日は[MASK]です。'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_fill_mask()