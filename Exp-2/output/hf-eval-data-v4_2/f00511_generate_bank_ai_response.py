# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_bank_ai_response(instruction, knowledge, dialog):
    """
    Generates a response for banking clients using a large-scale pre-trained dialog model.
    
    Args:
        instruction (str): The instruction or query from the banking client.
        knowledge (str): Additional knowledge to provide context for the response generation.
        dialog (list[str]): The dialog history to maintain context over a sequence of requests.
    
    Returns:
        str: The generated response for the client.
    
    Raises:
        ValueError: If input types are not as expected.
    """
    if not isinstance(instruction, str) or not isinstance(knowledge, str) or not isinstance(dialog, list):
        raise ValueError("Expected string for 'instruction', string for 'knowledge' and list of strings for 'dialog'.")

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

def test_generate_bank_ai_response():
    print("Testing started.")
    # Assume dummy data for testing
    instruction = "Check my account balance."
    knowledge = "The client has a savings account with a balance of $5,000."
    dialog_history = ["Welcome to the bank AI assistant. How may I help you today?"]

    # First test case: Valid inputs
    print("Testing case [1/3] started.")
    response = generate_bank_ai_response(instruction, knowledge, dialog_history)
    assert isinstance(response, str), f"Test case [1/3] failed: Expected a string response, got {type(response)}"

    # Second test case: Invalid instruction type
    print("Testing case [2/3] started.")
    try:
        generate_bank_ai_response(123, knowledge, dialog_history)
    except ValueError as e:
        assert str(e) == "Expected string for 'instruction', string for 'knowledge' and list of strings for 'dialog'.", f"Test case [2/3] failed: {e}"

    # Third test case: Empty dialog history should not raise an error
    print("Testing case [3/3] started.")
    response = generate_bank_ai_response(instruction, knowledge, [])
    assert isinstance(response, str), "Test case [3/3] failed: Expected a string response for empty dialog history."

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_bank_ai_response()