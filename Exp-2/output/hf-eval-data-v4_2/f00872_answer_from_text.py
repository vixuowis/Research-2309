# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_from_text(context, question):
    """
    Answers a question based on the given context using a pre-trained model.

    Args:
        context (str): The text from which the answer is to be extracted.
        question (str): The question to be answered.

    Returns:
        str: The answer extracted from the context.

    Raises:
        ValueError: If either context or question is None or an empty string.
    """
    if context is None or context.strip() == '':
        raise ValueError('The context should not be None or empty.')
    if question is None or question.strip() == '':
        raise ValueError('The question should not be None or empty.')

    # Initialize the question answering pipeline
    question_answerer = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

    # Use the pipeline to find the answer
    result = question_answerer(question=question, context=context)
    return result['answer']

# test_function_code --------------------

def test_answer_from_text():
    print("Testing started.")
    context = "SQuAD dataset consists of questions posed by crowdworkers on a set of Wikipedia articles."
    question = "What does SQuAD stand for?"

    # Testing case 1: Checking successful extraction of an answer
    print("Testing case [1/2] started.")
    answer = answer_from_text(context, question)
    assert answer.lower() == 'stanford question answering dataset', "Test case [1/2] failed: The model did not return the correct answer."

    # Testing case 2: Expecting ValueError for empty question
    print("Testing case [2/2] started.")
    try:
        answer_from_text(context, '')
        assert False, "Test case [2/2] failed: ValueError was not raised for an empty question."
    except ValueError:
        pass  # Expected behavior
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_from_text()