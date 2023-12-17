# requirements_file --------------------

!pip install -U transformers onnx

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_mars_orbit_duration(context: str) -> dict:
    """Get the duration of Mars' orbit around the sun using a question-answering model.

    Args:
        context (str): The context regarding Mars' orbit.

    Returns:
        dict: The answer provided by the question-answering pipeline.
    """
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    question = 'How long does it take for Mars to orbit the sun?'
    answer = qa_pipeline({'context': context, 'question': question})
    return answer

# test_function_code --------------------

def test_get_mars_orbit_duration():
    print("Testing started.")
    context = 'Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System, being larger than only Mercury. Mars takes approximately 687 Earth days to complete one orbit around the Sun.'

    # Test case 1: Check if the function returns a non-empty dictionary
    print("Testing case [1/1] started.")
    answer = get_mars_orbit_duration(context)
    assert isinstance(answer, dict) and answer, f"Test case [1/1] failed: Expected a dictionary with the answer, but got {answer}"
    print("Testing finished.")

# call_test_function_line --------------------

test_get_mars_orbit_duration()