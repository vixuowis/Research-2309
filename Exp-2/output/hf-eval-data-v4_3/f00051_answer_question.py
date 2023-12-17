# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModelForQuestionAnswering, pipeline

# function_code --------------------

def answer_question(context: str, question: str) -> str:
    """
    Answers the given question based on the provided context.

    Args:
        context (str): The context or passage where the answer may be found.
        question (str): The question we seek to find an answer for.

    Returns:
        str: The answer to the question, extracted from the context.

    Raises:
        ValueError: If the context or question is empty.
    """
    if not context or not question:
        raise ValueError('Context and question must not be empty.')

    model_name = 'deepset/roberta-base-squad2-distilled'
    qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline('question-answering', model=qa_model)
    result = qa_pipeline({'context': context, 'question': question})
    return result['answer']

# test_function_code --------------------

def test_answer_question():
    print("Testing started.")

    # Context and Question sample for testing
    context = "The Mona Lisa is a 16th century oil painting created by Leonardo. It's held at the Louvre in Paris."
    question = "Where is the Mona Lisa held?"

    # Test cases
    print("Testing case [1/1] started.")
    try:
        answer = answer_question(context, question)
        assert answer == 'the Louvre in Paris', f"Test case [1/1] failed: expected 'the Louvre in Paris', got {answer}"
    except Exception as e:
        print(f"Test case [1/1] failed with exception: {e}")

    print("Testing finished.")

# call_test_function_line --------------------

test_answer_question()