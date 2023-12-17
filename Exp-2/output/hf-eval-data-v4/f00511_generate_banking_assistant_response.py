# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_banking_assistant_response(instruction, knowledge, dialog):
    tokenizer = AutoTokenizer.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')

    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)

    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(query, return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# test_function_code --------------------

def test_generate_banking_assistant_response():
    print("Testing started.")
    # Example dialog
    dialog = ["Hello, how can I assist you today?"]

    # Test case 1: Customer wants to check account balance
    response = generate_banking_assistant_response('I want to check my balance.', '', dialog)
    assert 'account balance' in response, f"Test case 1 failed: {response}"

    # Test case 2: Customer needs information about loan options
    response = generate_banking_assistant_response('Tell me about loan options.', '', dialog)
    assert 'loan options' in response, f"Test case 2 failed: {response}"

    # Test case 3: Customer asks about branch locations
    response = generate_banking_assistant_response('Where is the nearest branch?', '', dialog)
    assert 'nearest branch' in response, f"Test case 3 failed: {response}"
    print("Testing finished.")

# Run the test function
test_generate_banking_assistant_response()