# function_import --------------------

from transformers import AutoTokenizer, AutoModelForMaskedLM

# function_code --------------------

def autoplay_fill_in_the_blank_game(text):
    """
    This function uses a pre-trained model 'bert-base-chinese' to predict the missing text in a fill-in-the-blank game.
    
    Args:
        text (str): The input text with blanks represented by '[MASK]'.
    
    Returns:
        str: The text with the blanks filled.
    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    model = AutoModelForMaskedLM.from_pretrained('bert-base-chinese')
    
    # Tokenize the input text
    input = tokenizer.encode(text, return_tensors='pt')
    
    # Predict the missing text
    mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
    token_logits = model(input)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    
    # Get the predicted token
    predicted_token = torch.argmax(mask_token_logits, dim=1)
    
    # Decode the token to get the predicted word
    predicted_word = tokenizer.decode(predicted_token)
    
    return text.replace('[MASK]', predicted_word)

# test_function_code --------------------

def test_autoplay_fill_in_the_blank_game():
    """
    This function tests the 'autoplay_fill_in_the_blank_game' function.
    """
    test_text = '我是[MASK]人'
    expected_result = '我是中国人'
    
    # Call the function with the test text
    result = autoplay_fill_in_the_blank_game(test_text)
    
    # Assert that the result is as expected
    assert result == expected_result, f'Expected {expected_result}, but got {result}'

# call_test_function_code --------------------

test_autoplay_fill_in_the_blank_game()