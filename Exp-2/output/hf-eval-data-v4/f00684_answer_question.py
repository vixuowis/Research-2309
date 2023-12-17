# requirements_file --------------------

!pip install -U transformers torch tensorflow

# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# function_code --------------------

def answer_question(question, context):
    """
    Answers a question based on the given context using a pre-trained ELECTRA language model.

    Parameters:
    - question (str): The question to be answered.
    - context (str): The context containing the information needed to answer the question.

    Returns:
    - str: The answer to the question.
    """
    # Load the pre-trained ELECTRA language model
    model = AutoModelForQuestionAnswering.from_pretrained('ahotrod/electra_large_discriminator_squad2_512')
    tokenizer = AutoTokenizer.from_pretrained('ahotrod/electra_large_discriminator_squad2_512')

    # Tokenize the question and context
    inputs = tokenizer(question, context, return_tensors='pt')
    outputs = model(**inputs)

    # Extract the start and end positions of the answer
    answer_start = outputs.start_logits.argmax().item()
    answer_end = outputs.end_logits.argmax().item() + 1

    # Convert token ids to the answer string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer

# test_function_code --------------------

def test_answer_question():
    print("Testing started.")

    # Test case 1: Known context and question
    question = "What is the capital of France?"
    context = "France is a country in Europe. Its capital is Paris."
    expected_answer = "Paris"

    print("Testing case [1/1] started.")
    actual_answer = answer_question(question, context)
    assert actual_answer == expected_answer, f"Test case [1/1] failed: Expected '{{expected_answer}}', got '{{actual_answer}}'"
    print("Testing case [1/1] successful.")

    print("Testing finished.")

# Run the test function
test_answer_question()