# function_import --------------------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# function_code --------------------

def chatbot_response(input_message: str) -> str:
    """
    This function takes a user's input message as a string and returns a response from a chatbot.
    The chatbot is powered by the 'facebook/blenderbot-1B-distill' model from Hugging Face Transformers.

    Args:
        input_message (str): The user's input message.

    Returns:
        str: The chatbot's response.
    """
    
    # Instantiate the model and tokenizer from the Hugging Face Transformers library
    chatbot = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-1B-distill")
    
    # Tokenize the input message
    chatbot_input_ids = tokenizer([input_message], return_tensors='pt')['input_ids']
    
    # Generate the chatbot's response
    chatbot_outputs = chatbot.generate(chatbot_input_ids)
    
    # Return only the text (without special tokens), strip off whitespace, and return to the user
    return tokenizer.batch_decode(chatbot_outputs)[0].strip()

# test_function_code --------------------

def test_chatbot_response():
    """
    This function tests the chatbot_response function with a few test cases.
    """
    assert chatbot_response('Hello, how are you?') != ''
    assert chatbot_response('What is the weather like today?') != ''
    assert chatbot_response('Tell me a joke.') != ''
    return 'All Tests Passed'


# call_test_function_code --------------------

test_chatbot_response()