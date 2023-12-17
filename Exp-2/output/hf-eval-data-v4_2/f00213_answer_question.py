# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_question(book_text: str, user_question: str) -> str:
    """
    Answers a question from a user reading a book using a pre-trained model.

    Args:
        book_text (str): The text from the book acting as the context for the question.
        user_question (str): The question asked by the user.

    Returns:
        str: The answer to the question as determined by the model.

    Raises:
        ValueError: If either the book_text or user_question is empty.
    """
    if not book_text or not user_question:
        raise ValueError("Both book_text and user_question must be provided.")
    qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2-distilled')
    result = qa_pipeline({'context': book_text, 'question': user_question})
    return result['answer']

# test_function_code --------------------

def test_answer_question():
    print("Testing started.")

    # Test case 1: Valid input
    print("Testing case [1/3] started.")
    book_text = "Once upon a time, there was a brave knight." 
    user_question = "Who was once upon a time?"
    assert answer_question(book_text, user_question) == "a brave knight", "Test case [1/3] failed: Incorrect answer."

    # Test case 2: Empty book_text
    print("Testing case [2/3] started.")
    try:
        answer_question('', 'What is the story about?')
        assert False, "Test case [2/3] failed: Exception not raised for empty book_text."
    except ValueError as e:
        assert str(e) == "Both book_text and user_question must be provided.", "Test case [2/3] failed: Incorrect exception message."

    # Test case 3: Empty user_question
    print("Testing case [3/3] started.")
    try:
        answer_question('It was a dark and stormy night.', '')
        assert False, "Test case [3/3] failed: Exception not raised for empty user_question."
    except ValueError as e:
        assert str(e) == "Both book_text and user_question must be provided.", "Test case [3/3] failed: Incorrect exception message."
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_question()