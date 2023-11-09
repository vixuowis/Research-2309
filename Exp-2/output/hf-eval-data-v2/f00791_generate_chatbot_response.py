# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_chatbot_response(instruction, knowledge, dialog):
    """
    Generate a response from a chatbot using the GODEL model.

    Args:
        instruction (str): The user's input.
        knowledge (str): Relevant external information.
        dialog (str): The previous dialog context.

    Returns:
        str: The generated output from the chatbot.
    """
    tokenizer = AutoTokenizer.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')

    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(query, return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

# test_function_code --------------------

def test_generate_chatbot_response():
    """
    Test the function generate_chatbot_response.
    """
    instruction = 'Tell me about rose gardening'
    knowledge = 'Roses need well-drained soil and plenty of sun.'
    dialog = ['Hello, how can I assist you with gardening today?']
    output = generate_chatbot_response(instruction, knowledge, dialog)
    assert isinstance(output, str), 'Output should be a string.'
    assert len(output) > 0, 'Output should not be empty.'

# call_test_function_code --------------------

test_generate_chatbot_response()