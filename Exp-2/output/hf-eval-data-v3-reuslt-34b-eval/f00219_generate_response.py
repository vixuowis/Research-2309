# function_import --------------------

from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# function_code --------------------

def generate_response(message: str) -> str:
    """
    Generate a response to a given message using the Blenderbot model.

    Args:
        message (str): The message to which the bot should respond.

    Returns:
        str: The bot's response to the message.
    """

    # load model and tokenizer from local cache or download from huggingface hub

    MODEL_NAME = "facebook/blenderbot-400M-distill"

    model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)

    tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)

    # encode message and return the corresponding logits

    inputs = tokenizer([message], return_tensors="pt")

    reply_ids = model.generate(**inputs)

    # use tokenizer to decode model output into a string

    response = "\n".join([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in reply_ids])
    
    return response

# test_function_code --------------------

def test_generate_response():
    """
    Test the generate_response function.
    """
    message1 = "How can I cancel my subscription?"
    message2 = "What is your return policy?"
    message3 = "Do you offer discounts on bulk orders?"
    assert isinstance(generate_response(message1), str)
    assert isinstance(generate_response(message2), str)
    assert isinstance(generate_response(message3), str)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_response()