# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer
import torch

# function_code --------------------

def answer_document_question(document_content, question):
    """
    Answer a question based on the content of a given document using a pre-trained model.

    Args:
        document_content (str): The content of the document to be analyzed.
        question (str): The question to answer based on the document content.

    Returns:
        str: The answer to the question derived from the document content.

    Raises:
        ValueError: If the document content or question is not provided.
    """
    if not document_content or not question:
        raise ValueError('The document content and question must be provided.')

    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    inputs = tokenizer(document_content, question, return_tensors='pt', padding='max_length', max_length=512, truncation='only_second')
    outputs = model(**inputs)
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer.strip()

# test_function_code --------------------

def test_answer_document_question():
    print("Testing started.")
    # Prepare a sample document and a question
    document_content = "The quick brown fox jumps over the lazy dog."
    question = "What does the fox do?"

    # Testing case 1
    print("Testing case [1/1] started.")
    answer = answer_document_question(document_content, question)
    assert answer == 'jumps', f"Test case [1/1] failed: Expected 'jumps', got '{answer}'"
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_document_question()