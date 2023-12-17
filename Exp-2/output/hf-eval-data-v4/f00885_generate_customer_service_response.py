# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_customer_service_response(instruction, knowledge, dialog):
    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')

    # Format the context and knowledge
    knowledge_prefixed = '[KNOWLEDGE] ' + knowledge
    dialog_joined = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog_joined} {knowledge_prefixed}"

    # Tokenize and generate response
    input_ids = tokenizer(query, return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)

    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# test_function_code --------------------

def test_generate_customer_service_response():
    print("Testing generate_customer_service_response function.")
    instruction = "How can I respond to a customer complaint about late delivery?"
    knowledge = "The courier had external delays due to bad winter weather."
    dialog = [
        "Customer: My package is late. What's going on?",
        "Support: I apologize for the inconvenience. I'll check what's happening with the package and get back to you."
    ]

    # Call the function to generate a response
    response = generate_customer_service_response(instruction, knowledge, dialog)
    assert isinstance(response, str), "The function should return a string."
    assert len(response) > 0, "The function should return a non-empty string."

    print("Test passed successfully.")

# Run the test
test_generate_customer_service_response()