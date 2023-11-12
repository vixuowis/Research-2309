# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer
import torch

# function_code --------------------

def extract_invoice_info(doc_text: str, question: str) -> str:
    """
    Extracts specific information from an invoice document using a pre-trained model.

    Args:
        doc_text (str): The text content of the document.
        question (str): The question that needs to be answered based on the document.

    Returns:
        str: The answer to the question based on the document.

    Raises:
        ImportError: If the required libraries are not installed.
    """
    # Load the pre-trained model and tokenizer
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')

    # Process the document and question with the tokenizer
    inputs = tokenizer(doc_text, question, return_tensors='pt')

    # Perform inference using the model
    outputs = model(**inputs)

    # Get the answer from the model's output
    answer = torch.argmax(outputs.logits)
    return tokenizer.decode(answer)

# test_function_code --------------------

def test_extract_invoice_info():
    """Tests the extract_invoice_info function."""
    doc_text = 'Invoice No: 12345\nDate: 01/01/2022\nTotal Amount: $100.00'
    question = 'What is the invoice number?'
    assert extract_invoice_info(doc_text, question) == '12345'
    question = 'What is the date of the invoice?'
    assert extract_invoice_info(doc_text, question) == '01/01/2022'
    question = 'What is the total amount of the invoice?'
    assert extract_invoice_info(doc_text, question) == '$100.00'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_invoice_info()