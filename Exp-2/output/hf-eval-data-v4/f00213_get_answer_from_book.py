# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer_from_book(book_text, user_question):
    """
    This function takes a passage of text from a book and a question posed by a user,
    then uses a pre-trained NLP model to find the answer within the text.

    Parameters:
        book_text (str): The text from the book to search within.
        user_question (str): The question asked by the user.

    Returns:
        str: The answer extracted from the book text as determined by the model.
    """
    qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2-distilled')
    result = qa_pipeline({'context': book_text, 'question': user_question})
    answer = result['answer']
    return answer

# test_function_code --------------------

def test_get_answer_from_book():
    print("Testing get_answer_from_book function.")
    book_text = "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do."
    user_question = "Who was getting tired?"

    # Expected answer is 'Alice'
    print("Testing case [1/1] started.")
    answer = get_answer_from_book(book_text, user_question)
    assert answer == 'Alice', f"Test case [1/1] failed: Expected 'Alice', got {answer}"
    print("Testing finished.")

test_get_answer_from_book()