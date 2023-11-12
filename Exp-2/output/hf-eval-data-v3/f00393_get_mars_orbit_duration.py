# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_mars_orbit_duration(context: str, question: str) -> str:
    '''
    This function uses the transformers pipeline to answer the question about Mars orbit duration.

    Args:
        context (str): The context about Mars orbit.
        question (str): The question to be answered.

    Returns:
        str: The answer to the question.

    Raises:
        ValueError: If the model cannot be loaded.
    '''
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    answer = qa_pipeline({'context': context, 'question': question})
    return answer['answer']

# test_function_code --------------------

def test_get_mars_orbit_duration():
    '''
    This function tests the get_mars_orbit_duration function.
    '''
    context = 'Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System, being larger than only Mercury. Mars takes approximately 687 Earth days to complete one orbit around the Sun.'
    question = 'How long does it take for Mars to orbit the sun?'
    answer = get_mars_orbit_duration(context, question)
    assert isinstance(answer, str), 'The answer should be a string.'
    assert '687' in answer, 'The answer should contain the number of days it takes for Mars to orbit the sun.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_mars_orbit_duration()