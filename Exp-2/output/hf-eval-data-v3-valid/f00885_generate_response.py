# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_response(instruction: str, knowledge: str, dialog: list) -> str:
    """
    Generate a response based on the instruction, knowledge, and dialog.

    Args:
        instruction (str): Instruction on how to respond.
        knowledge (str): Knowledge about the situation.
        dialog (list): List of dialogues.

    Returns:
        str: Generated response.
    """
    tokenizer = AutoTokenizer.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f'{instruction} [CONTEXT] {dialog} {knowledge}'
    input_ids = tokenizer(query, return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

# test_function_code --------------------

def test_generate_response():
    """
    Test the generate_response function.
    """
    instruction = 'How can I respond to a customer complaint about late delivery?'
    knowledge = 'The courier had external delays due to bad winter weather.'
    dialog = ['Customer: My package is late. What is going on?', 'Support: I apologize for the inconvenience. I will check what is happening with the package and get back to you.']
    response = generate_response(instruction, knowledge, dialog)
    assert isinstance(response, str), 'The response should be a string.'
    assert len(response) > 0, 'The response should not be empty.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_response()