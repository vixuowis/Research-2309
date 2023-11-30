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

    # Load the pretrained BlenderBot-1b model from Hugging Face Transformers 
    # and a tokenizer that can be used to convert between text inputs and numerical representations.
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-1B-distill")

    # Encode the user's input message using the loaded tokenizer 
    # and append the end of sequence (EOS) token ID to it.
    input_ids = tokenizer.encode(input_message + tokenizer.eos_token, return_tensors='pt')
    
    # Generate chatbot responses using the loaded model. 
    chatbot_outputs = model.generate(
        input_ids, max_length=500, pad_token_id=tokenizer.pad_token_id)
    
    # Convert the chatbot output to text using the tokenizer's decode method.
    response_text =  tokenizer.decode(chatbot_outputs[0], skip_special_tokens=True)  
    
    return response_text

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