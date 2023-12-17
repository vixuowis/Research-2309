# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch", "datasets", "tokenizers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_information(context, question):
    """
    Extracts relevant answers to specific questions from OCR-scanned text using a pre-trained model.

    Args:
        context (str): The OCR extracted text from the document.
        question (str): The question whose answer is to be extracted from the given context.

    Returns:
        dict: A dictionary with the question and extracted answer.

    Raises:
        ValueError: If the context or question is empty.

    """
    if not context or not question:
        raise ValueError('The context and question must not be empty.')

    # Initialize the question-answering pipeline with the specified model
    qa_pipeline = pipeline('question-answering', model='tiennvcs/layoutlmv2-large-uncased-finetuned-vi-infovqa')

    # Use the pipeline to extract the answer from the context based on the question
    result = qa_pipeline({"context": context, "question": question})
    return result

# test_function_code --------------------

def test_extract_information():
    print("Testing started.")
    context = "This is a test context containing various information for extraction."
    questions = [
        "What information is extracted?",
        "Is this a test context?"
    ]

    # Testing case 1
    print("Testing case [1/2] started.")
    answer1 = extract_information(context, questions[0])
    assert answer1['score'] > 0, "Test case [1/2] failed: No answer extracted."

    # Testing case 2
    print("Testing case [2/2] started.")
    answer2 = extract_information(context, questions[1])
    assert answer2['score'] > 0, "Test case [2/2] failed: No answer extracted."
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_information()