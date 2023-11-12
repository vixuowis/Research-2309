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
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/blenderbot-1B-distill')
    tokenizer = AutoTokenizer.from_pretrained('facebook/blenderbot-1B-distill')
    inputs = tokenizer(input_message, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'])
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

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