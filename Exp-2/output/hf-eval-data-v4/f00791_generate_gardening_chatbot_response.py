# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_gardening_chatbot_response(instruction, knowledge, dialog):
    tokenizer = AutoTokenizer.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')

    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    if dialog:
        dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(query, return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output


# test_function_code --------------------

def test_generate_gardening_chatbot_response():
    print("Testing generate_gardening_chatbot_response function.")

    # Test case 1: Check if the response generated is a non-empty string
    print("Testing case [1/1] started.")
    sample_instruction = 'How do I prune my roses?'
    sample_knowledge = 'Pruning roses encourages growth and increases bloom.'
    sample_dialog = []
    response = generate_gardening_chatbot_response(sample_instruction, sample_knowledge, sample_dialog)
    assert isinstance(response, str) and response, "Test case [1/1] failed: The response should be a non-empty string."
    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_generate_gardening_chatbot_response()
