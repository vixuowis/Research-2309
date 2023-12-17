# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer

# function_code --------------------

def respond_to_smart_home_message(input_message: str) -> str:
    """
    Respond to a message intended for a smart home system.

    Args:
        input_message (str): The message from the user to the smart home system.

    Returns:
        str: The response message from the smart home system.

    Raises:
        ValueError: If the input message is not a string.
    """
    if not isinstance(input_message, str):
        raise ValueError('Input message must be a string.')

    tokenizer = AutoTokenizer.from_pretrained('facebook/blenderbot-90M')
    model = AutoModelForCausalLM.from_pretrained('facebook/blenderbot-90M')
    tokenized_input = tokenizer.encode(input_message + tokenizer.eos_token, return_tensors='pt')
    output = model.generate(tokenized_input, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, tokenized_input.shape[-1]:][0], skip_special_tokens=True)
    return response

# test_function_code --------------------

def test_respond_to_smart_home_message():
    print('Testing started.')

    # Test case 1
    print('Testing case [1/3] started.')
    response = respond_to_smart_home_message('Turn off the lights.')
    assert isinstance(response, str), 'Test case [1/3] failed: Response is not a string.'

    # Test case 2
    print('Testing case [2/3] started.')
    try:
        respond_to_smart_home_message(123)
    except ValueError as e:
        assert str(e) == 'Input message must be a string.', 'Test case [2/3] failed: Did not raise ValueError for non-string input.'

    # Test case 3
    print('Testing case [3/3] started.')
    response = respond_to_smart_home_message('What is the temperature inside?')
    assert 'degree' in response or 'temperature' in response, 'Test case [3/3] failed: Response does not contain expected keywords.'
    print('Testing finished.')

# call_test_function_line --------------------

test_respond_to_smart_home_message()