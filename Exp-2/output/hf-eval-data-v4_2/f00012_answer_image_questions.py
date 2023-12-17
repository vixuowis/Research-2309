# requirements_file --------------------

!pip install -U transformers Pillow

# function_import --------------------

from transformers import pipeline
from PIL import Image

# function_code --------------------

def answer_image_questions(image_path: str, question: str) -> dict:
    """
    Answers questions about an image using a pretrained model.

    Args:
        image_path (str): The file path to the image to be analyzed.
        question (str): The question to ask about the image.

    Returns:
        dict: Contains the answer to the question about the image.

    Raises:
        FileNotFoundError: If the image_path does not lead to a valid file.
        ValueError: If the question is empty or not a string.
    """
    # Check if the image path is valid
    try:
        image = Image.open(image_path)
    except IOError:
        raise FileNotFoundError('The provided image path does not lead to a valid file.')

    # Check if the question is valid
    if not question or not isinstance(question, str):
        raise ValueError('The question must be a non-empty string.')

    # Load the visual question answering model
    vqa = pipeline('visual-question-answering', model='microsoft/git-base-vqav2')

    # Get the answer to the question
    answer = vqa(image=image, question=question)
    return answer

# test_function_code --------------------

def test_answer_image_questions():
    print("Testing started.")
    sample_image_path = 'sample.jpg'  # Replace with path to a test image
    sample_question = 'What is in the image?'

    # Testing case 1: Valid image and question
    print("Testing case [1/2] started.")
    try:
        result = answer_image_questions(sample_image_path, sample_question)
        assert isinstance(result, dict), f"Test case [1/2] failed: Expected dict, got {type(result)}"
    except Exception as e:
        assert False, f"Test case [1/2] failed: {e}"

    # Testing case 2: Invalid image path
    print("Testing case [2/2] started.")
    try:
        answer_image_questions('invalid.jpg', sample_question)
        assert False, "Test case [2/2] failed: FileNotFoundError was expected."
    except FileNotFoundError:
        pass  # Expected exception
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_image_questions()