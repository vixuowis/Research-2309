# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def get_financial_report_summary(question, context):
    """
    Processes the financial report to extract answer to the question provided.

    Args:
        question (str): The question related to financial report.
        context (str): The financial report content.

    Returns:
        str: The answer to the question from the financial report.

    Raises:
        ValueError: If the model prediction is invalid or cannot fetch the answer.
    """
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')
    tokenizer = AutoTokenizer.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')

    inputs = tokenizer(question, context, return_tensors='pt')
    output = model(**inputs)
    start_position = output.start_logits.argmax().item()
    end_position = output.end_logits.argmax().item()
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_position:end_position + 1]))
    if not answer:
        raise ValueError('Unable to fetch the answer from the financial report.')
    return answer

# test_function_code --------------------

def test_get_financial_report_summary():
    print("Testing started.")

    # Test case 1: Check if the function provides correct answer for known input.
    print("Testing case [1/2] started.")
    question = "What were the total revenues for the last quarter?"
    context = "In the last quarter, the company's total revenues were reported at $3.2 million with a gross profit of $1.5 million. The operating expenses during the same quarter were $1 million."
    answer = get_financial_report_summary(question, context)
    assert answer == "$3.2 million", f"Test case [1/2] failed: Expected $3.2 million, got {answer}"

    # Test case 2: Validate response to invalid input.
    print("Testing case [2/2] started.")
    question = "What is the net profit?"
    context = ""
    try:
        get_financial_report_summary(question, context)
        assert False, "Test case [2/2] failed: Function should raise a ValueError with invalid input."
    except ValueError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_get_financial_report_summary()