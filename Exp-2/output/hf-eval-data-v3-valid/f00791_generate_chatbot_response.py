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
    tokenizer = AutoTokenizer.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')

    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f'{instruction} [CONTEXT] {dialog} {knowledge}'
    input_ids = tokenizer(query, return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

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