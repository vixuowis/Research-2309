# function_import --------------------

from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# function_code --------------------

def generate_response(user_input):
    """
    This function uses the BlenderbotForConditionalGeneration model from Hugging Face's transformers library
    to generate a response to a given user input.

    Args:
        user_input (str): The user's input message.

    Returns:
        str: The model's generated response.
    """
    model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-3B')
    tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-3B')

    inputs = tokenizer([user_input], return_tensors='pt')
    outputs = model.generate(**inputs)
    reply = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return reply

# test_function_code --------------------

def test_generate_response():
    """
    This function tests the generate_response function by providing a sample user input and checking
    if the output is of the correct type (str).
    """
    user_input = 'What are the benefits of regular exercise?'
    output = generate_response(user_input)

    assert isinstance(output, str), 'Output should be a string.'

# call_test_function_code --------------------

test_generate_response()