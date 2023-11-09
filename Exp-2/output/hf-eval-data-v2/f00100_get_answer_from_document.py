# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def get_answer_from_document(document_text: str, question_text: str) -> str:
    """
    This function uses a pre-trained model to answer questions based on a given document.

    Args:
        document_text (str): The text of the document to be processed.
        question_text (str): The question to be answered based on the document.

    Returns:
        str: The answer to the question based on the document.
    """
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    inputs = tokenizer(document_text, question_text, return_tensors='pt')
    outputs = model(**inputs)
    answer_start = outputs.start_logits.argmax(dim=-1).item()
    answer_end = outputs.end_logits.argmax(dim=-1).item() + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer

# test_function_code --------------------

def test_get_answer_from_document():
    """
    This function tests the get_answer_from_document function.
    It uses a sample document and question to test the function.
    """
    document_text = 'This is a sample document. It contains information about various topics.'
    question_text = 'What does the document contain?'
    answer = get_answer_from_document(document_text, question_text)
    assert isinstance(answer, str), 'The function should return a string.'
    assert answer != '', 'The function should return a non-empty string.'

# call_test_function_code --------------------

test_get_answer_from_document()