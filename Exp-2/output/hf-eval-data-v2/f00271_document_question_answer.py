# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer

# function_code --------------------

def document_question_answer(document_content: str, question: str) -> str:
    """
    This function uses a pre-trained model from Hugging Face Transformers to answer questions based on the content of a given document.

    Args:
        document_content (str): The content of the document.
        question (str): The question to be answered.

    Returns:
        str: The answer to the question based on the document content.
    """
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    inputs = tokenizer(document_content, question, return_tensors='pt', padding='max_length', max_length=512, truncation='only_first')
    outputs = model(**inputs)
    answer = tokenizer.decode(outputs[0])
    return answer

# test_function_code --------------------

def test_document_question_answer():
    """
    This function tests the document_question_answer function.
    It uses a sample document and question, and asserts that the returned answer is not None.
    """
    document_content = 'This is a sample document. It contains information about various topics.'
    question = 'What does the document contain?'
    answer = document_question_answer(document_content, question)
    assert answer is not None, 'The function did not return an answer.'

# call_test_function_code --------------------

test_document_question_answer()