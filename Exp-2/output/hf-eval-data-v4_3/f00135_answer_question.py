# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_question(context, question):
    """
    Answer a question based on the given context using a pretrained transformer model.

    Args:
        context (str): The context paragraph or content where the answer may be found.
        question (str): The question for which the answer is sought.

    Returns:
        dict: A dictionary with the answer, score, start index, and end index.

    Raises:
        ValueError: If either context or question is an empty string.
    """
    if not context or not question:
        raise ValueError('Context and question must be provided.')
    
    # Initialize the question-answering pipeline with the pretrained model
    qa_pipeline = pipeline('question-answering', model='csarron/bert-base-uncased-squad-v1', tokenizer='csarron/bert-base-uncased-squad-v1')
    
    # Execute the pipeline and return the result
    return qa_pipeline({'context': context, 'question': question})

# test_function_code --------------------

def test_answer_question():
    print("Testing started.")
    # Define test context and question
    context = "The game was played on February 7, 2016 at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."
    question = "What day was the game played on?"

    # Testing case 1: Valid input
    print("Testing case [1/1] started.")
    result = answer_question(context, question)
    assert 'answer' in result, f"Test case [1/1] failed: 'answer' key not found in result"
    assert result['answer'] == 'February 7, 2016', f"Test case [1/1] failed: Expected 'February 7, 2016', got {result['answer']}"
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_question()