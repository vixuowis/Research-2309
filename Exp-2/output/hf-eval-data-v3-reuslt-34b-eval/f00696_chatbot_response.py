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

    # Load the HF model and tokenizer objects
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-1B-distill")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")

    # Tokenize the user input message and use the chatbot model to predict a response
    inputs = tokenizer(input_message, return_tensors="pt")
    reply_ids = model.generate(**inputs)
    
    # Decode the predicted response's tokens using the tokenizer
    response = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in reply_ids][0]

    return response


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