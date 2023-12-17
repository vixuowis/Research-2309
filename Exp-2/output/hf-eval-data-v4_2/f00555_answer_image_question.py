# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_image_question(image_path, question):
    """
    Answers a question based on the content of an image.

    Args:
        image_path (str): The path to the image file.
        question (str): The question related to the image content.

    Returns:
        str: The answer to the question based on the image.

    Raises:
        FileNotFoundError: If the image_path does not correspond to a file.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    vqa = pipeline('visual-question-answering', model='JosephusCheung/GuanacoVQAOnConsumerHardware')
    return vqa(image_path, question)

# test_function_code --------------------

def test_answer_image_question():
    print("Testing started.")

    # Assuming test_image.jpg exists in the same directory with a known content
    image_path = 'test_image.jpg'
    question = 'What is the main color of the object?'
    expected_answer = 'blue'  # Expected answer for the test

    # Testing case 1
    print("Testing case [1/1] started.")
    actual_answer = answer_image_question(image_path, question)
    assert actual_answer == expected_answer, f"Test case [1/1] failed: Expected {expected_answer}, got {actual_answer}"
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_image_question()