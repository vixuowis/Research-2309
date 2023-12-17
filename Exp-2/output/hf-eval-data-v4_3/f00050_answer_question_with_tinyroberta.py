# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def answer_question_with_tinyroberta(question, context):
    """
    This function utilizes the 'deepset/tiny-roberta-squad2' model from HuggingFace's Transformers
    to answer a question given a context.

    Args:
        question (str): The question to answer.
        context (str): The context containing information to answer the question.

    Returns:
        dict: A dictionary with the answer, its score, start and end positions in the context.

    Raises:
        ValueError: If inputs are not strings or are empty.
    """
    # Validate input types and content
    if not isinstance(question, str) or not isinstance(context, str):
        raise ValueError('Inputs must be strings.')
    if not question or not context:
        raise ValueError('Inputs must not be empty.')

    # Initialize model and tokenizer
    model_name = 'deepset/tiny-roberta-squad2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Create a pipeline for question answering
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    
    # Formulate the input and get the result
    QA_input = {'question': question, 'context': context}
    result = nlp(QA_input)
    return result

# test_function_code --------------------

def test_answer_question_with_tinyroberta():
    print('Testing started.')
    
    # Test case: regular question and context
    print('Testing case [1/1] started.')
    question = 'Why is model conversion important?'
    context = 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    result = answer_question_with_tinyroberta(question, context)
    assert isinstance(result, dict), 'Result should be a dictionary.'
    assert 'answer' in result, 'Result dictionary should contain an answer key.'
    assert len(result['answer']) > 0, 'The answer should not be empty.'
    print('Testing finished.')

# call_test_function_line --------------------

test_answer_question_with_tinyroberta()