# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer

# function_code --------------------

def extract_total_amount(question: str, context: str) -> str:
    """
    Extracts the total amount from an invoice document using a pre-trained model.

    Args:
        question (str): The question to be answered by the model. In this case, 'What is the total amount?'.
        context (str): The context in which the question is to be answered. This is the invoice document.

    Returns:
        str: The total amount due from the invoice document.
    """
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('impira/layoutlm-invoices')
    tokenizer = AutoTokenizer.from_pretrained('impira/layoutlm-invoices')
    inputs = tokenizer(question, context, return_tensors='pt')
    outputs = model(**inputs)
    answer_start = outputs.start_logits.argmax().item()
    answer_end = outputs.end_logits.argmax().item()
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start: answer_end + 1].tolist()))
    return answer

# test_function_code --------------------

def test_extract_total_amount():
    """
    Tests the extract_total_amount function.
    """
    question = 'What is the total amount?'
    context = 'Invoice information for order ABC_123\nProduct: Widget A, Quantity: 10, Price: $5 each\nProduct: Widget B, Quantity: 5, Price: $3 each\nProduct: Widget C, Quantity: 15, Price: $2 each\nSubtotal: $75, Tax: $6.38, Total Amount Due: $81.38'
    expected_answer = '$81.38'
    assert extract_total_amount(question, context) == expected_answer

# call_test_function_code --------------------

test_extract_total_amount()