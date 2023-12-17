# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

# function_code --------------------

def extract_answers(question, context):
    """
    Extracts the answer to a question from a given context using a fine-tuned DeBERTa-v3 model.

    Args:
        question (str): The question to be answered.
        context (str): The context within which the answer will be searched.

    Returns:
        str: The extracted answer.

    Raises:
        ValueError: If either question or context is not provided.

    """
    if not question or not context:
        raise ValueError("Both question and context must be provided.")
    model_name = 'deepset/deberta-v3-large-squad2'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    QA_input = {'question': question, 'context': context}
    answer = nlp(QA_input)
    return answer['answer']

# test_function_code --------------------

def test_extract_answers():
    print("Testing started.")
    # Test case 1: question and context provided
    print("Testing case [1/2] started.")
    question = 'What are the benefits of exercise?'
    context = 'Exercise helps maintain a healthy body weight, improves cardiovascular health, and boosts the immune system.'
    expected_answer = 'maintain a healthy body weight, improves cardiovascular health, and boosts the immune system'
    assert extract_answers(question, context) == expected_answer, f"Test case [1/2] failed: expected '{expected_answer}', got '{extract_answers(question, context)}'"

    # Test case 2: One or both parameters are empty strings
    print("Testing case [2/2] started.")
    try:
        extract_answers('', '')
        assert False, "Test case [2/2] failed: ValueError not raised when expected."
    except ValueError as e:
        assert str(e) == "Both question and context must be provided.", f"Test case [2/2] failed: expected ValueError with message 'Both question and context must be provided.', got '{str(e)}'"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_answers()