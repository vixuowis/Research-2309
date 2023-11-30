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
    
    # Initialize the model
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/DialoGPT-large")
    except OSError as e:
        print(e)
        
    # Create the dialog context from the dialog history (i.e., previous chats)
    if len(dialog):
        chat_history_ids = [50256] + tokenizer.encode(" ".join(dialog[:-1])) + [tokenizer.sep_token_id]
    else:
        chat_history_ids = []
        
    # Prepare inputs for generating responses
    input_ids = bytearray(chat_history_ids + tokenizer.encode(instruction, add_special_tokens=False) + [50256])
    input_ids = torch.tensor([input_ids]).unsqueeze(0)
    
    # Generate the response based on the dialog context
    with torch.no_grad():
        out = model.generate(input_ids, max_length=len(instruction)+150, pad_token_id=50256)
        
    return tokenizer.decode(out[0], skip_special_tokens=True).replace("##", "")

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