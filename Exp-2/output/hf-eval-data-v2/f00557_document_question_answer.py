# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer

# function_code --------------------

def document_question_answer(question: str, scanned_document_text: str) -> str:
    """
    This function uses a pre-trained model from Hugging Face Transformers to answer questions based on a scanned document.

    Args:
        question (str): The question to be answered.
        scanned_document_text (str): The text extracted from the scanned document.

    Returns:
        str: The answer to the question.
    """
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    inputs = tokenizer(question, scanned_document_text, return_tensors='pt')
    output = model(**inputs)
    return output

# test_function_code --------------------

def test_document_question_answer():
    """
    This function tests the document_question_answer function.
    It uses a sample question and a sample scanned document text.
    """
    question = 'What is the title of the document?'
    scanned_document_text = 'This is a sample document. The title of the document is Sample Document.'
    answer = document_question_answer(question, scanned_document_text)
    assert isinstance(answer, str), 'The output should be a string.'

# call_test_function_code --------------------

test_document_question_answer()