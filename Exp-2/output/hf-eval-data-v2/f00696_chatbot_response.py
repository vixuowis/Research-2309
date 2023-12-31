# function_import --------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# function_code --------------------

def chatbot_response(input_message):
    """
    This function takes a string as input and returns a response generated by a conversational chatbot.
    The chatbot is powered by the 'facebook/blenderbot-1B-distill' model from Hugging Face Transformers.

    Args:
        input_message (str): The user's input message that the chatbot should respond to.

    Returns:
        str: The chatbot's response to the input message.
    """
    # Load the pre-trained model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/blenderbot-1B-distill')
    tokenizer = AutoTokenizer.from_pretrained('facebook/blenderbot-1B-distill')

    # Encode the user's input message
    inputs = tokenizer(input_message, return_tensors='pt')

    # Generate a response using the model
    outputs = model.generate(inputs['input_ids'])

    # Decode the generated output, skipping special tokens
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded_output

# test_function_code --------------------

def test_chatbot_response():
    """
    This function tests the 'chatbot_response' function by providing a sample input message and printing the chatbot's response.
    """
    # Define a sample input message
    input_message = 'Hello, how are you?'

    # Call the 'chatbot_response' function with the sample input message
    response = chatbot_response(input_message)

    # Print the chatbot's response
    print(f'Chatbot response: {response}')

    # Assert that the response is not None or an empty string
    assert response is not None and response != '', 'The chatbot response is None or an empty string.'

# call_test_function_code --------------------

test_chatbot_response()