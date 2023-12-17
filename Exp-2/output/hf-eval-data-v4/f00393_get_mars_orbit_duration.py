# requirements_file --------------------

!pip install -U transformers onnx

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_mars_orbit_duration():
    """
    This function uses the transformers pipeline for question-answering to determine
    how long it takes for Mars to orbit the sun.
    
    Returns:
        str: The duration of Mars' orbit around the sun in Earth days.
    """
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    context = 'Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System, being larger than only Mercury. Mars takes approximately 687 Earth days to complete one orbit around the Sun.'
    question = 'How long does it take for Mars to orbit the sun?'
    answer = qa_pipeline({'context': context, 'question': question})
    return answer['answer']

# test_function_code --------------------

def test_get_mars_orbit_duration():
    print("Testing get_mars_orbit_duration function.")
    duration = get_mars_orbit_duration()
    expected_duration = '687 Earth days'
    assert duration == expected_duration, f"Test failed: Expected {expected_duration}, got {duration}"
    print("Test passed.")

# Execute the test function
test_get_mars_orbit_duration()