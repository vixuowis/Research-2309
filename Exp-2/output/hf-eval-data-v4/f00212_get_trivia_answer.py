# requirements_file --------------------

!pip install -U transformers onnx

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_trivia_answer(context, question):
    """
    This function takes a historical context and a trivia question, then uses a DistilBERT-based
    model to answer the question based on the context.

    Parameters:
    context (str): Historical information relevant to the trivia question.
    question (str): The trivia question to be answered.

    Returns:
    str: The answer to the trivia question as determined by the model.
    """
    # Load the question-answering pipeline with the specified model
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    # Use the pipeline to get the answer
    answer = qa_pipeline({'context': context, 'question': question})
    # Return the answer provided by the model
    return answer['answer']

# test_function_code --------------------

def test_get_trivia_answer():
    print("Testing get_trivia_answer function.")
    context = 'In 1492, Christopher Columbus sailed the ocean blue, discovering the New World.'
    question = 'Who discovered the New World?'
    expected_answer = 'Christopher Columbus'

    # Test case 1
    print("Testing case [1/1] started.")
    actual_answer = get_trivia_answer(context, question)
    assert actual_answer == expected_answer, f'Test case [1/1] failed: expected {expected_answer}, but got {actual_answer}'
    print("Testing case [1/1] succeeded.")

    print("Testing finished.")