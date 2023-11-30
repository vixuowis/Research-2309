# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_chatbot_response(instruction, knowledge, dialog):
    """
    Generate a chatbot response based on the instruction, knowledge, and dialog.

    Args:
        instruction (str): The user's input.
        knowledge (str): Relevant external information.
        dialog (list): The previous dialog context.

    Returns:
        str: The generated output from the chatbot.

    Raises:
        OSError: If there is an error in loading the model or tokenizer.
    """

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/DialoGPT-medium")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    except OSError as error:
        raise OSError(f"Error in loading the chatbot's model or tokenizer - {error}")

    input_text = " ".join([instruction, knowledge, *dialog])

    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")

    response = model.generate(input_ids=input_ids, max_length=1000)

    output_text = tokenizer.decode(response[0], skip_special_tokens=True)
    
    start = output_text.find(">>") + 2
    end = output_text.find("\n", start)

    return output_text[start:end].strip()

# test_function_code --------------------

def test_generate_chatbot_response():
    """
    Test the generate_chatbot_response function.
    """
    instruction = 'Tell me about roses'
    knowledge = 'Roses are a type of flowering shrub.'
    dialog = ['Hello, how can I help you today?', 'I want to know about roses.']
    output = generate_chatbot_response(instruction, knowledge, dialog)
    assert isinstance(output, str), 'Output should be a string'

    instruction = 'How to plant a rose?'
    knowledge = 'To plant a rose, you need to...'
    dialog = ['Hello, how can I help you today?', 'I want to plant a rose.']
    output = generate_chatbot_response(instruction, knowledge, dialog)
    assert isinstance(output, str), 'Output should be a string'

    instruction = 'What is the best time to plant roses?'
    knowledge = 'The best time to plant roses is...'
    dialog = ['Hello, how can I help you today?', 'When should I plant roses?']
    output = generate_chatbot_response(instruction, knowledge, dialog)
    assert isinstance(output, str), 'Output should be a string'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_chatbot_response()