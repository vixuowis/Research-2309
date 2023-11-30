# function_import --------------------

from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# function_code --------------------

def generate_response(user_input: str) -> str:
    """
    Generate a response to the user input using the BlenderbotForConditionalGeneration model.

    Args:
        user_input (str): The user's input message.

    Returns:
        str: The model's response.

    Raises:
        OSError: If there is a problem with the model loading or the disk quota is exceeded.
    """

    try:
        model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    except OSError:
        raise OSError(
            """
            There was a problem with loading the model.
            Please ensure your API call includes the correct endpoint URL and parameters, then try again.
            """
        )

    # Tokenize and prepare the user's input message.
    history = tokenizer(user_input, return_tensors="pt")

    # Generate a response using the model.
    reply_ids = model.generate(**history)
    output = list(tokenizer.batch_decode(reply_ids, skip_special_tokens=True))[0]
    
    return output


# test_function_code --------------------

def test_generate_response():
    """
    Test the generate_response function.
    """
    user_input = 'What are the benefits of regular exercise?'
    output = generate_response(user_input)
    assert isinstance(output, str), 'The output should be a string.'

    user_input = 'Tell me a joke.'
    output = generate_response(user_input)
    assert isinstance(output, str), 'The output should be a string.'

    user_input = 'What is the weather like today?'
    output = generate_response(user_input)
    assert isinstance(output, str), 'The output should be a string.'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_response()