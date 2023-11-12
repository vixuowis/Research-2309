# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def get_financial_report_summary(question: str, context: str) -> str:
    '''
    This function uses a pre-trained model from Hugging Face Transformers to answer questions based on a given context.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the answer is to be found.

    Returns:
        str: The answer to the question based on the context.
    '''
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')
    tokenizer = AutoTokenizer.from_pretrained('L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V18_08_04_2023')
    inputs = tokenizer(question, context, return_tensors='pt')
    output = model(**inputs)
    start_position = output.start_logits.argmax().item()
    end_position = output.end_logits.argmax().item()
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_position:end_position + 1]))
    return answer

# test_function_code --------------------

def test_get_financial_report_summary():
    '''
    This function tests the get_financial_report_summary function.
    '''
    question = 'What were the total revenues for the last quarter?'
    context = 'In the last quarter, the company's total revenues were reported at $3.2 million with a gross profit of $1.5 million. The operating expenses during the same quarter were $1 million.'
    assert get_financial_report_summary(question, context) == '$3.2 million'
    question = 'What were the operating expenses during the same quarter?'
    assert get_financial_report_summary(question, context) == '$1 million'
    question = 'What was the gross profit?'
    assert get_financial_report_summary(question, context) == '$1.5 million'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_financial_report_summary()