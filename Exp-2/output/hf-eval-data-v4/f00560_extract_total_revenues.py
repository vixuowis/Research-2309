# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def extract_total_revenues(question, context):
    # Load the pre-trained model and the tokenizer
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')
    tokenizer = AutoTokenizer.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')

    # Tokenize the question and the context
    inputs = tokenizer(question, context, return_tensors='pt')
    output = model(**inputs)

    # Extract the answer from the output
    start_position = output.start_logits.argmax().item()
    end_position = output.end_logits.argmax().item()
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_position:end_position + 1]))
    return answer

# test_function_code --------------------

def test_extract_total_revenues():
    print("Testing started.")
    # Define the question and context based on the test case
    question = "What were the total revenues for the last quarter?"
    context = "In the last quarter, the company's total revenues were reported at $3.2 million with a gross profit of $1.5 million. The operating expenses during the same quarter were $1 million."

    # Expected answer
    expected_answer = "$3.2 million"

    # Run the function and compare with the expected answer
    print("Testing case [1/1] started.")
    real_answer = extract_total_revenues(question, context)
    assert real_answer == expected_answer, f"Test case failed: Expected {expected_answer}, got {real_answer}"
    print("Testing finished. All cases passed.")

# Run the test function
test_extract_total_revenues()