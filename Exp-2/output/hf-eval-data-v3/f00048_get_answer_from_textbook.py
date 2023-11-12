# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer_from_textbook(question: str, textbook_content: str) -> str:
    """
    This function uses a pre-trained model from the transformers library to answer questions based on the provided textbook content.

    Args:
        question (str): The question to be answered.
        textbook_content (str): The textbook content to find the answer from.

    Returns:
        str: The answer to the question.
    """
    qa_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
    result = qa_model(question=question, context=textbook_content)
    return result['answer']

# test_function_code --------------------

def test_get_answer_from_textbook():
    """
    This function tests the get_answer_from_textbook function.
    """
    question1 = 'What is the function of mitochondria in a cell?'
    textbook_content1 = 'Mitochondria are the energy factories of the cell. They convert energy from food molecules into a useable form known as adenosine triphosphate (ATP).'
    assert get_answer_from_textbook(question1, textbook_content1) == 'the energy factories of the cell'

    question2 = 'What is the capital of France?'
    textbook_content2 = 'The capital of France is Paris.'
    assert get_answer_from_textbook(question2, textbook_content2) == 'Paris'

    question3 = 'Who was the first president of the United States?'
    textbook_content3 = 'The first president of the United States was George Washington.'
    assert get_answer_from_textbook(question3, textbook_content3) == 'George Washington'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_answer_from_textbook()