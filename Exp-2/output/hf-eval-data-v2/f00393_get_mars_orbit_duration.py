# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_mars_orbit_duration(context: str, question: str) -> str:
    '''
    This function uses the 'philschmid/distilbert-onnx' model from the transformers library to answer the question about the duration of Mars' orbit around the sun.

    Args:
        context (str): The context about Mars' orbit.
        question (str): The question to be answered.

    Returns:
        str: The answer to the question.
    '''
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    answer = qa_pipeline({'context': context, 'question': question})
    return answer['answer']

# test_function_code --------------------

def test_get_mars_orbit_duration():
    '''
    This function tests the 'get_mars_orbit_duration' function by providing a context and a question.
    '''
    context = 'Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System, being larger than only Mercury. Mars takes approximately 687 Earth days to complete one orbit around the Sun.'
    question = 'How long does it take for Mars to orbit the sun?'
    answer = get_mars_orbit_duration(context, question)
    assert '687' in answer, 'Test failed!'
    print('Test passed!')

# call_test_function_code --------------------

test_get_mars_orbit_duration()