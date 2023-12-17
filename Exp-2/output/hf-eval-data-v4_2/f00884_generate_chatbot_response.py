# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# function_code --------------------

def generate_chatbot_response(user_input):
    '''
    Generates a response from the chatbot based on the user's input.

    Args:
        user_input (str): The input message from the user.

    Returns:
        str: The chatbot's generated response.

    Raises:
        ValueError: If user_input is not a string.
    '''
    if not isinstance(user_input, str):
        raise ValueError('user_input must be a string.')

    model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-3B')
    tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-3B')

    inputs = tokenizer([user_input], return_tensors='pt')
    outputs = model.generate(**inputs)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return response

# test_function_code --------------------

def test_generate_chatbot_response():
    print("Testing started.")
    # Assuming 'BlenderbotForConditionalGeneration' and 'BlenderbotTokenizer' are available.

    # Test case 1: Normal input
    print("Testing case [1/1] started.")
    input_message = "What are the benefits of regular exercise?"
    response = generate_chatbot_response(input_message)
    assert isinstance(response, str), f"Test case [1/1] failed: Expected string response, got {type(response)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_chatbot_response()