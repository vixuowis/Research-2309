# requirements_file --------------------

import subprocess

requirements = ["transformers", "sentencepiece"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_medical_question(context: str, question: str) -> str:
    """
    Answers a medical question based on the provided context using a pre-trained NLP model.

    Args:
        context (str): The context containing information relevant to the question.
        question (str): The medical question to be answered.

    Returns:
        str: The answer to the given question.

    Raises:
        ValueError: If any of the inputs are empty strings.
    """
    if not context or not question:
        raise ValueError('The context and question must not be empty.')
    qa_pipeline = pipeline('question-answering', model='sultan/BioM-ELECTRA-Large-SQuAD2')
    result = qa_pipeline({'context': context, 'question': question})
    return result['answer']

# test_function_code --------------------

def test_answer_medical_question():
    print("Testing started.")

    # Test case 1: Valid context and question
    print("Testing case [1/1] started.")
    context = "Pneumonia is an infection that inflames the air sacs in one or both lungs."
    question = "What is pneumonia?"
    try:
        answer = answer_medical_question(context, question)
        assert answer == 'an infection that inflames the air sacs in one or both lungs', f"Test case [1/1] failed: Expected 'an infection that inflames the air sacs in one or both lungs', got {answer}"
    except ValueError as e:
        assert False, f"Test case [1/1] failed with ValueError: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_answer_medical_question()