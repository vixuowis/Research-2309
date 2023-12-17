# requirements_file --------------------

!pip install -U transformers torch datasets tokenizers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_medical_info(document_text, question):
    """
    Extracts medical information from a document based on a question.

    Args:
        document_text (str): The text of the medical document to be analyzed.
        question (str): The question to find an answer for within the document text.

    Returns:
        dict: A dictionary with the answer and additional information extracted by the model.

    Raises:
        ValueError: If document_text or question is empty.
    """
    if not document_text or not question:
        raise ValueError('Document text and question must not be empty.')

    qa_pipeline = pipeline('question-answering', model='bigwiz83/sapbert-from-pubmedbert-squad2')
    return qa_pipeline({'context': document_text, 'question': question})

# test_function_code --------------------

def test_extract_medical_info():
    print('Testing started.')
    # A hypothetical piece of text from a medical document and a question.
    document_text = 'Atrial fibrillation is a common type of arrhythmia that can lead to blood clots, stroke, heart failure, and other heart-related complications.'
    question = 'What complications can atrial fibrillation lead to?'

    # Testing case 1: Expected outcome with valid inputs
    print('Testing case [1/1] started.')
    try:
        answer = extract_medical_info(document_text, question)
        assert 'answer' in answer, f'Test case [1/1] failed: Answer is not found in the result.'
    except ValueError as e:
        assert False, f'Test case [1/1] failed with Value Error: {e}'
    print('Testing finished.')



# call_test_function_line --------------------

test_extract_medical_info()