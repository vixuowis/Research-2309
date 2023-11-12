# function_import --------------------

from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer

# function_code --------------------

def predict_next_word(phrase: str) -> str:
    """
    Predicts the next word in a given phrase using the DebertaV2ForMaskedLM model.

    Args:
        phrase (str): The phrase to predict the next word for. The phrase should end with '<|mask|>'.

    Returns:
        str: The predicted next word.

    Raises:
        ValueError: If the phrase does not end with '<|mask|>'.
    """
    if not phrase.endswith('<|mask|>'):
        raise ValueError('Phrase should end with "<|mask|>".')

    mask_model = DebertaV2ForMaskedLM.from_pretrained('microsoft/deberta-v2-xxlarge')
    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xxlarge')

    processed = tokenizer(phrase, return_tensors='pt')
    predictions = mask_model(**processed).logits.argmax(dim=-1)

    return tokenizer.decode(predictions[0], skip_special_tokens=True)

# test_function_code --------------------

def test_predict_next_word():
    """
    Tests the predict_next_word function.
    """
    test_phrase_1 = 'The dog jumped over the <|mask|>'
    test_phrase_2 = 'She went to the <|mask|>'
    test_phrase_3 = 'I am a <|mask|>'

    assert isinstance(predict_next_word(test_phrase_1), str)
    assert isinstance(predict_next_word(test_phrase_2), str)
    assert isinstance(predict_next_word(test_phrase_3), str)

    try:
        predict_next_word('This phrase does not end with mask')
    except ValueError:
        pass
    else:
        raise AssertionError('ValueError not raised for phrase not ending with "<|mask|>".')

    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_next_word()