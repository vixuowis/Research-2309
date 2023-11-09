# function_import --------------------

from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer

# function_code --------------------

def predict_next_word(phrase: str) -> str:
    """
    Predicts the next word in a given phrase using the DebertaV2ForMaskedLM model.

    Args:
        phrase (str): The phrase to predict the next word for. The phrase should end with '<|mask|>'.

    Returns:
        str: The predicted word.
    """
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
    test_phrase = 'The dog jumped over the <|mask|>'
    predicted_word = predict_next_word(test_phrase)
    assert isinstance(predicted_word, str), 'The predicted word should be a string.'
    assert predicted_word, 'The predicted word should not be an empty string.'

# call_test_function_code --------------------

test_predict_next_word()