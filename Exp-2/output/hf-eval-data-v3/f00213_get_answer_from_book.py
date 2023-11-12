# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer_from_book(book_text: str, user_question: str) -> str:
    '''
    This function uses a pre-trained model from Hugging Face Transformers to answer questions based on a given context.

    Args:
        book_text (str): The context in which to find the answer.
        user_question (str): The question to answer.

    Returns:
        str: The answer to the question.
    '''
    qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2-distilled')
    result = qa_pipeline({'context': book_text, 'question': user_question})
    return result['answer']

# test_function_code --------------------

def test_get_answer_from_book():
    '''
    This function tests the get_answer_from_book function.
    '''
    book_text = 'The sky is blue.'
    user_question = 'What color is the sky?'
    assert get_answer_from_book(book_text, user_question) == 'blue'
    
    book_text = 'Dogs are mammals.'
    user_question = 'What are dogs?'
    assert get_answer_from_book(book_text, user_question) == 'mammals'
    
    book_text = 'The earth revolves around the sun.'
    user_question = 'What does the earth revolve around?'
    assert get_answer_from_book(book_text, user_question) == 'the sun'
    
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_answer_from_book()