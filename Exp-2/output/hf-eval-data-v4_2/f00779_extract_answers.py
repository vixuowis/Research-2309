# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_answers(question, context):
    """
    Extracts the answer to a question from a given context using a pre-trained transformer model.

    Args:
        question (str): The question to be answered.
        context (str): The context containing the potential answer.

    Returns:
        dict: The answer found by the model.

    Raises:
        ValueError: If either question or context is not provided.
    """
    if not question or not context:
        raise ValueError('Question and context must be provided.')

    nlp = pipeline('question-answering', model='deepset/deberta-v3-large-squad2', tokenizer='deepset/deberta-v3-large-squad2')
    QA_input = {'question': question, 'context': context}
    result = nlp(QA_input)
    return result

# test_function_code --------------------

def test_extract_answers():
    print("Testing started.")
    sample_context = 'The penalty for breaking the contract includes financial restitution and possible legal action.'

    # Test case 1: Valid question and context
    print("Testing case [1/2] started.")
    result1 = extract_answers('What does breaking the contract entail?', sample_context)
    assert 'financial restitution' in result1['answer'], f"Test case [1/2] failed: {result1}"

    # Test case 2: Missing context
    print("Testing case [2/2] started.")
    try:
        extract_answers('What does breaking the contract entail?', '')
        assert False, 'Test case [2/2] failed: Function should raise ValueError for missing context'
    except ValueError as e:
        assert str(e) == 'Question and context must be provided.', f"Test case [2/2] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_answers()