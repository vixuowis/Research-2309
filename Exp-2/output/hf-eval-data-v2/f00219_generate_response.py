# function_import --------------------

from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# function_code --------------------

def generate_response(message):
    """
    This function uses the BlenderbotForConditionalGeneration model to generate a response to a given message.

    Args:
        message (str): The message to which the bot should respond.

    Returns:
        str: The generated response.
    """
    model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')
    tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
    inputs = tokenizer(message, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# test_function_code --------------------

def test_generate_response():
    """
    This function tests the generate_response function by providing a sample message and checking the type of the response.
    """
    message = "How can I cancel my subscription?"
    response = generate_response(message)
    assert isinstance(response, str), "The response should be a string."

# call_test_function_code --------------------

test_generate_response()