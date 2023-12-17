# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# function_code --------------------

def extract_legal_information(contract_text, question):
    """
    Extract answers from a legal contract text for a given question using a pretrained model.

    Args:
        contract_text (str): The legal contract text.
        question (str): The question to get an answer for.

    Returns:
        str: The answer extracted from the contract text.

    Raises:
        ValueError: If either contract_text or question is empty.
    """
    # Validate input arguments
    if not contract_text or not question:
        raise ValueError('The contract_text and question cannot be empty.')

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('Rakib/roberta-base-on-cuad')
    model = AutoModelForQuestionAnswering.from_pretrained('Rakib/roberta-base-on-cuad')

    # Tokenize the inputs
    inputs = tokenizer(question, contract_text, return_tensors='pt')
    # Get model output
    outputs = model(**inputs)
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

    # Get the most likely start and end of answer
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

    return answer.strip()

# test_function_code --------------------

def test_extract_legal_information():
    print('Testing started.')
    sample_contract = "We hereby grant the Licensee the exclusive right to develop, construct, operate and promote the Project, as well as to manage the daily operations of the Licensed Facilities during the Term."
    sample_question = "What rights are granted to the Licensee?"

    # Test case 1: Check for non-empty answer
    print('Testing case [1/1] started.')
    answer = extract_legal_information(sample_contract, sample_question)
    assert answer, f'Test case [1/1] failed: Expected a non-empty answer.'
    print('Testing finished.')

# call_test_function_line --------------------

test_extract_legal_information()