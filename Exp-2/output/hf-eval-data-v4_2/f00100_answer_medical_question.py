# requirements_file --------------------

!pip install -U transformers==4.12.2 torch==1.8.0+cu101

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering
import torch

# function_code --------------------

def answer_medical_question(document_text, question_text):
    """
    Answers a patient-related question based on their medical document using a pretrained NLP model.

    Args:
        document_text (str): The text of the medical document.
        question_text (str): The question related to the medical document.

    Returns:
        str: The answer extracted from the document.

    Raises:
        ValueError: If the document_text or question_text is not provided.

    """
    if not document_text or not question_text:
        raise ValueError('The document text and question text must be provided.')

    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')

    inputs = tokenizer(document_text, question_text, return_tensors='pt')
    outputs = model(**inputs)

    answer_start = outputs.start_logits.argmax(dim=-1).item()
    answer_end = outputs.end_logits.argmax(dim=-1).item() + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer

# test_function_code --------------------

def test_answer_medical_question():
    print("Testing started.")
    document_text = 'Patient has a history of hypertension and diabetes.'
    question_text = 'What is the patient's history?'

    # Test case 1: Check if the function returns the correct type.
    print("Testing case [1/2] started.")
    answer = answer_medical_question(document_text, question_text)
    assert isinstance(answer, str), f"Test case [1/2] failed: Expected str, got {type(answer)}"

    # Test case 2: Check if the function raises ValueError for empty input.
    print("Testing case [2/2] started.")
    try:
        _ = answer_medical_question('', '')
        assert False, 'Test case [2/2] failed: ValueError not raised for empty inputs.'
    except ValueError as e:
        assert str(e) == 'The document text and question text must be provided.', f"Test case [2/2] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_answer_medical_question()