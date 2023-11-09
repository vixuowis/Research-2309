# function_import --------------------

from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# function_code --------------------

def get_chatbot_response(input_text):
    """
    This function uses the BlenderbotForConditionalGeneration model from Hugging Face to generate a response to a given input text.
    
    Args:
        input_text (str): The input text that the chatbot should respond to.
    
    Returns:
        str: The chatbot's response to the input text.
    """
    model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot_small-90M')
    tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot_small-90M')
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# test_function_code --------------------

def test_get_chatbot_response():
    """
    This function tests the get_chatbot_response function by providing it with a sample input text and checking if the output is a string.
    """
    input_text = 'What is the admission process for the new academic year?'
    response = get_chatbot_response(input_text)
    assert isinstance(response, str), 'The output should be a string.'

# call_test_function_code --------------------

test_get_chatbot_response()