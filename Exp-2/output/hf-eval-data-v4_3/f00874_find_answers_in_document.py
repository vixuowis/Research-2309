# requirements_file --------------------

import subprocess

requirements = ["torch", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def find_answers_in_document(context, question):
    """
    Finds answers to specified questions within a given document.

    Args:
        context (str): The document's text where the answer will be searched for.
        question (str): The question for which the answer is sought.

    Returns:
        dict: A dictionary containing the answer and the start and end position.

    Raises:
        ValueError: If either context or question is an empty string.
    """
    if not context or not question:
        raise ValueError('The context and question should not be empty.')

    qa_pipeline = pipeline('question-answering', model='bert-large-cased-whole-word-masking-finetuned-squad')
    return qa_pipeline({'context': context, 'question': question})


# test_function_code --------------------

def test_find_answers_in_document():
    print("Testing started.")
    context = "This is a long document containing information about AI research, breakthroughs and projects."
    questions = [
            "What does the document contain?",
            "What are the breakthroughs mentioned?",
            "What projects are included in the document?"
        ]

    for i, question in enumerate(questions):
        print(f"Testing case [{i+1}/{len(questions)}] started.")
        answer = find_answers_in_document(context, question)
        assert 'answer' in answer, f"Test case [{i+1}/{len(questions)}] failed: 'answer' key not found in the result."
    print("Testing finished.")


# call_test_function_line --------------------

test_find_answers_in_document()