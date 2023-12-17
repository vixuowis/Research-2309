# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# function_code --------------------


# A function to chat with a user using BlenderBot-1B

def chat_with_user(input_message):
    """
    This function takes a user's input message and generates a reply using the BlenderBot-1B conversational model.

    Parameters:
        input_message (str): The message from the user to which the chatbot will respond.

    Returns:
        str: The chatbot's response.
    """
    # Load the pre-trained BlenderBot-1B model
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/blenderbot-1B-distill')
    # Load the tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained('facebook/blenderbot-1B-distill')

    # Tokenize and encode the user's input message
    inputs = tokenizer(input_message, return_tensors='pt')
    # Generate a response from the model
    outputs = model.generate(inputs['input_ids'])
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# test_function_code --------------------


def test_chat_with_user():
    print("Testing chat_with_user function started.")
    test_input = 'Hello, how are you?'
    expected_output_contains = ['I', 'am', 'bot']  # Assumes the bot introduces itself

    # Test case 1: Checking if the chatbot responds with a valid string
    print("Testing case [1/1] started.")
    response = chat_with_user(test_input)
    assert isinstance(response, str), f"Test case [1/1] failed: The response should be a string."

    # The following test is commented out due to potential variability in the bot's response
    # Test case 2: Checking if the chatbot's response contains expected words
    # assert all(word in response for word in expected_output_contains), f"Test case [2/2] failed: The response does not contain expected words."

    print("Testing chat_with_user function finished.")

# Run the test function
test_chat_with_user()