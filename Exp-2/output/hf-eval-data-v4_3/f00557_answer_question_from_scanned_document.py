# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer
import torch

# function_code --------------------

def answer_question_from_scanned_document(question, document_text):
    """
    Answers a question based on a scanned document text.

    Args:
        question (str): The question to be answered.
        document_text (str): The text extracted from the scanned document.

    Returns:
        str: The answer to the question extracted from the document.

    Raises:
        ValueError: If the inputs are not strings or are empty.
    """
    if not isinstance(question, str) or not isinstance(document_text, str):
        raise ValueError('The `question` and `document_text` arguments must be strings.')
    if not question or not document_text:
        raise ValueError('The `question` and `document_text` arguments cannot be empty.')

    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')

    inputs = tokenizer(question, document_text, return_tensors='pt')
    outputs = model(**inputs)

    # Assuming that the answer is in the output tokens
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

    return answer

# test_function_code --------------------

def test_answer_question_from_scanned_document():
    print("Testing started.")
    # Assuming we have a function `load_sample_scanned_document` that provides a sample document and its associated text.
    document_text = load_sample_scanned_document()
    question = 'What is the date on the document?'

    # Testing case 1: Proper input
    print("Testing case [1/2] started.")
    answer = answer_question_from_scanned_document(question, document_text)
    assert answer, f"Test case [1/2] failed: answer should not be empty"

    # Testing case 2: Improper input
    print("Testing case [2/2] started.")
    try:
        answer_question_from_scanned_document('', '')
        assert False, "Test case [2/2] failed: ValueError expected for empty input"
    except ValueError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_answer_question_from_scanned_document()