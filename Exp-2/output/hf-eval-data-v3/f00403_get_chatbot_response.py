# function_import --------------------

from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# function_code --------------------

def get_chatbot_response(input_text: str) -> str:
    """
    This function uses the Hugging Face's Blenderbot model to generate a response to a given input text.

    Args:
        input_text (str): The input text to which the chatbot should respond.

    Returns:
        str: The generated response from the chatbot.

    Raises:
        KeyError: If there is an issue with tokenization process.
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
    This function tests the get_chatbot_response function with various test cases.
    """
    assert isinstance(get_chatbot_response('What is the admission process for the new academic year?'), str)
    assert isinstance(get_chatbot_response('Who are the teachers for the science department?'), str)
    assert isinstance(get_chatbot_response('What extracurricular activities are available?'), str)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_chatbot_response()