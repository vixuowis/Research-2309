# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer(question: str, context: str) -> str:
    """
    This function uses the Hugging Face Transformers library to answer questions based on a given context.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the answer is to be found.

    Returns:
        str: The answer to the question.

    Raises:
        OSError: If there is a problem with the disk quota.
    """
    
    # Instantiate QA pipeline. See https://huggingface.co/transformers/main_classes/pipelines.html#question-answering for documentation.
    qa_pipeline = pipeline("question-answering")
    # Call pipeline on question and context.
    result = qa_pipeline({'question': question, 'context': context})
    return result['answer']

# test_function_code --------------------

def test_get_answer():
    """
    This function tests the get_answer function with a few test cases.
    """
    assert get_answer('What is the capital of Germany?', 'Berlin is the capital of Germany.') == 'Berlin'
    assert get_answer('Who won the world cup in 2018?', 'France won the world cup in 2018.') == 'France'
    assert get_answer('What is the color of the sky?', 'The sky is blue.') == 'blue'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_get_answer()