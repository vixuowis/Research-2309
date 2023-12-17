# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_question_from_text(context, question):
    """
    Answers a question given a text using DistilBERT model.

    Args:
        context (str): The text from which the answer should be extracted.
        question (str): The question to be answered.

    Returns:
        str: The extracted answer.
    """
    question_answerer = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
    result = question_answerer(question=question, context=context)
    return result['answer']

# test_function_code --------------------

def test_answer_question_from_text():
    print("Testing started.")

    # Test case 1: Expecting an answer from provided context and question
    context = "Transformers provides state-of-the-art machine learning models for natural language processing tasks."
    question = "What does Transformers provide?"
    expected_answer = "state-of-the-art machine learning models"
    print("Testing case [1/1] started.")
    answer = answer_question_from_text(context, question)
    assert answer == expected_answer, f"Test case [1/1] failed: Expected '{{expected_answer}}', but got '{{answer}}'."
    print("Testing finished.")

# Running the test function
test_answer_question_from_text()