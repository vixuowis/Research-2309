# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer

# function_code --------------------

def extract_total_amount(question, context):
    # Load the pre-trained model and tokenizer
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('impira/layoutlm-invoices')
    tokenizer = AutoTokenizer.from_pretrained('impira/layoutlm-invoices')

    # Tokenize the input question and context
    inputs = tokenizer(question, context, return_tensors='pt')

    # Obtain the model's output
    outputs = model(**inputs)

    # Get the start and end positions of the answer
    answer_start = outputs.start_logits.argmax().item()
    answer_end = outputs.end_logits.argmax().item()

    # Decode the tokens to get the answer
    answer = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end + 1])
    return answer

# test_function_code --------------------

def test_extract_total_amount():
    print('Testing extract_total_amount function...')
    question = 'What is the total amount?'
    context = 'Invoice information for order ABC_123\nProduct: Widget A, Quantity: 10, Price: $5 each\nProduct: Widget B, Quantity: 5, Price: $3 each\nProduct: Widget C, Quantity: 15, Price: $2 each\nSubtotal: $75, Tax: $6.38, Total Amount Due: $81.38'
    expected_answer = '$81.38'

    answer = extract_total_amount(question, context)
    assert answer == expected_answer, f'Expected {expected_answer}, but got {answer}.'

    print('All tests passed!')