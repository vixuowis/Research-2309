# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def multilingual_question_answering(context: str, question: str) -> str:
    """
    Answers a question based on the provided context using a multilingual BERT model.

    Args:
        context (str): The text content from which to find the answer.
        question (str): The question to be answered.

    Returns:
        str: The answer to the question based on the context.

    Raises:
        ValueError: If either 'context' or 'question' is not provided or empty.

    """
    if not context or not question:
        raise ValueError("'context' and 'question' cannot be empty.")
    qa_pipeline = pipeline('question-answering', model='mrm8488/bert-multi-cased-finetuned-xquadv1', tokenizer='mrm8488/bert-multi-cased-finetuned-xquadv1')
    answer = qa_pipeline({'context': context, 'question': question})
    return answer['answer']

# test_function_code --------------------

def test_multilingual_question_answering():
    print("Testing started.")
    # Assume these are sample inputs for testing
    sample_context = "Ceci est un exemple de texte d'affaires en français."
    sample_questions = [
        "Quelle est la couleur du ciel ?",
        "Où se trouve la Tour Eiffel ?"
    ]
    expected_answers = [
        None, # expected to not find an answer
        'Paris' # expected answer (hypothetical)
    ]

    for i, (question, expected) in enumerate(zip(sample_questions, expected_answers), 1):
        print(f"Testing case [{i}/{len(sample_questions)}] started.")
        answer = multilingual_question_answering(sample_context, question)
        assert answer == expected, f"Test case [{i}/{len(sample_questions)}] failed: Expected {expected}, got {answer}"
    print("Testing finished.")

# call_test_function_line --------------------

test_multilingual_question_answering()