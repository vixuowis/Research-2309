# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_gardening_chatbot_response(instruction, knowledge, dialog):
    """Generate a response for a gardening chatbot based on instruction, external knowledge, and dialog context.

    Args:
        instruction (str): The user's input instruction.
        knowledge (str): The relevant external knowledge.
        dialog (list of str): The previous dialog context.

    Returns:
        str: The generated response from the chatbot.

    Raises:
        ValueError: If the inputs are not properly formatted.
    """
    if not isinstance(dialog, list) or not isinstance(instruction, str) or not isinstance(knowledge, str):
        raise ValueError("Inputs must be a string for instruction and knowledge, and a list of strings for dialog.")
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    tokenizer = AutoTokenizer.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    input_ids = tokenizer(query, return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

# test_function_code --------------------

def test_generate_gardening_chatbot_response():
    print("Testing started.")
    instruction = "How to prune a rose bush?"
    knowledge = "Cut back old wood about 30 to 40 percent before growth begins."
    dialog = ["Welcome to Gardening Chat! How can I assist you?"]

    # Test case 1: Verify valid inputs generate a response
    print("Testing case [1/1] started.")
    response = generate_gardening_chatbot_response(instruction, knowledge, dialog)
    assert isinstance(response, str), f"Test case [1/1] failed: The function should return a string response."
    print("Testing finished.")


# call_test_function_line --------------------

test_generate_gardening_chatbot_response()