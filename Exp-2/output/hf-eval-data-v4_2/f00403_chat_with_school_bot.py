# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# function_code --------------------

def chat_with_school_bot(message: str) -> str:
    """
    Generates a response from the school chatbot for a given message.

    Args:
        message (str): The query or message sent to the school chatbot.

    Returns:
        str: The chatbot's response to the input message.

    Raises:
        ValueError: If the input message is empty.
    """
    if not message:
        raise ValueError('The input message is empty.')

    # Initialize the chatbot model and tokenizer
    model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot_small-90M')
    tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot_small-90M')

    # Tokenize the input message for the chatbot
    inputs = tokenizer(message, return_tensors='pt')

    # Generate a response using the model
    outputs = model.generate(**inputs)

    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# test_function_code --------------------

def test_chat_with_school_bot():
    print("Testing started.")

    # Test case 1: Query about admission process
    print("Testing case [1/3] started.")
    admission_query = "What is the admission process for the new academic year?"
    admission_response = chat_with_school_bot(admission_query)
    assert isinstance(admission_response, str), f"Test case [1/3] failed: Expected a string response, got {type(admission_response)}"

    # Test case 2: Empty message handling
    print("Testing case [2/3] started.")
    try:
        empty_response = chat_with_school_bot("")
        assert False, "Test case [2/3] failed: Expected ValueError for empty message"
    except ValueError as e:
        assert str(e) == 'The input message is empty.', f"Test case [2/3] failed: {e}"

    # Test case 3: Query about extracurricular activities
    print("Testing case [3/3] started.")
    activities_query = "Can you tell me about extracurricular activities?"
    activities_response = chat_with_school_bot(activities_query)
    assert isinstance(activities_response, str), f"Test case [3/3] failed: Expected a string response, got {type(activities_response)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_chat_with_school_bot()