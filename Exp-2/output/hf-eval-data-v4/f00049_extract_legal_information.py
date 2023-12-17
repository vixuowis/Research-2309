# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# function_code --------------------

def extract_legal_information(contract_text, question):
    """
    Extracts information from a legal contract using a pre-trained NLP model.

    Args:
        contract_text (str): The text of the legal contract.
        question (str): The question to be answered based on the contract.

    Returns:
        str: The answer extracted from the contract.
    """
    # Load the pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('Rakib/roberta-base-on-cuad')
    model = AutoModelForQuestionAnswering.from_pretrained('Rakib/roberta-base-on-cuad')

    # Tokenize the input text
    inputs = tokenizer(question, contract_text, return_tensors='pt', truncation=True)

    # Perform the question answering
    outputs = model(**inputs)

    # Extract the answer
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

    return answer

# test_function_code --------------------

def test_extract_legal_information():
    print("Testing started.")
    # Define sample contract text
    contract_text = "The Licensee shall pay to the Licensor the full amount of Ten Million (10,000,000) Dollars."
    # Define question
    question = "How much does the Licensee need to pay?"

    # Perform the test
    print("Testing case [1/1] started.")
    answer = extract_legal_information(contract_text, question)
    assert answer == 'Ten Million (10,000,000) Dollars', f"Test case failed: Incorrect answer {answer}"
    print("Testing finished. Successful.")

# Run the test function
test_extract_legal_information()