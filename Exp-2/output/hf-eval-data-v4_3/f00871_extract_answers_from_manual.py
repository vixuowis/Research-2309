# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline, AutoModel, AutoTokenizer

# function_code --------------------

def extract_answers_from_manual(question, manual_content):
    """
    Extracts answers to a given question from a provided product manual content
    using a pre-trained BERT model.

    Args:
        question (str): The question for which the answer is sought.
        manual_content (str): The content of the product manual.

    Returns:
        str: The extracted answer from the given context if available, otherwise 'Answer not found'.

    Raises:
        ValueError: If any of the inputs are empty strings.
    """
    if not question or not manual_content:
        raise ValueError('Question and manual content must not be empty.')

    qa_pipeline = pipeline(
        'question-answering',
        model=AutoModel.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2'),
        tokenizer=AutoTokenizer.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2')
    )

    input_data = {'question': question, 'context': manual_content}
    answer = qa_pipeline(input_data)
    return answer['answer'] if answer['score'] > 0.1 else 'Answer not found'

# test_function_code --------------------

def test_extract_answers_from_manual():
    print("Testing started.")
    manual_content = "The product can be reset to factory settings by pressing the reset button for 5 seconds."
    question = "How to perform a factory reset on the product?"

    # Testing case [1/1] started
    print("Testing case [1/1] started.")
    answer = extract_answers_from_manual(question, manual_content)
    assert answer == 'by pressing the reset button for 5 seconds', f"Test case [1/1] failed: Expected 'by pressing the reset button for 5 seconds', got {answer}"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_answers_from_manual()