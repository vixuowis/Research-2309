# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer

# function_code --------------------

def extract_invoice_info(doc_text, question):
    """
    Extract specific information from an invoice document.

    Args:
        doc_text (str): The text content of the document.
        question (str): The question that needs to be answered from the document.

    Returns:
        str: The answer to the question from the document.
    """
    # Load the pretrained model and tokenizer
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')

    # Process the document and question with the tokenizer
    inputs = tokenizer(doc_text, question, return_tensors='pt')

    # Perform inference using the model
    outputs = model(**inputs)

    # Extract the answer from the outputs
    answer = tokenizer.decode(outputs['answer_start'][0], outputs['answer_end'][0])

    return answer

# test_function_code --------------------

def test_extract_invoice_info():
    """
    Test the function extract_invoice_info.
    """
    # Define a sample document and question
    doc_text = 'Invoice No: 12345\nDate: 01/01/2022\nTotal Amount: $100'
    question = 'What is the invoice number?'

    # Call the function with the sample document and question
    answer = extract_invoice_info(doc_text, question)

    # Assert the answer is correct
    assert answer == '12345', f'Expected 12345, but got {answer}'

# call_test_function_code --------------------

test_extract_invoice_info()