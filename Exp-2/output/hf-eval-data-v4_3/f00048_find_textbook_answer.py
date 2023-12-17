# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def find_textbook_answer(question: str, context: str) -> str:
    """
    Finds the answer to a given question within a specific textbook context using a fine-tuned BERT model.

    Args:
        question (str): The question to be answered.
        context (str): The textbook content serving as the context for the answer.

    Returns:
        str: The extracted answer from the context.

    Raises:
        ValueError: If the question or context is empty.
    """
    if not question or not context:
        raise ValueError('Question and context cannot be empty.')

    qa_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
    result = qa_model(question=question, context=context)
    return result['answer']

# test_function_code --------------------

def test_find_textbook_answer():
    print("Testing started.")
    # Assuming question and context are provided correctly
    question = "What is the purpose of photosynthesis?"
    context = "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy."

    # Test case 1
    print("Testing case [1/1] started.")
    answer = find_textbook_answer(question, context)
    assert answer == 'convert light energy into chemical energy', f"Test case [1/1] failed: Expected 'convert light energy into chemical energy', got '{answer}'"
    print("Testing finished.")

# call_test_function_line --------------------

test_find_textbook_answer()