# requirements_file --------------------

!pip install -U transformers Pillow

# function_import --------------------

from transformers import LayoutLMv3ForQuestionAnswering, LayoutLMv3Tokenizer
from PIL import Image

# function_code --------------------

def answer_questions_from_form_image(image_path, question):
    """Uses a pre-trained LayoutLMv3 model to answer a question based on the content of a form image.

    Args:
        image_path (str): The path to the image file of the enrollment form.
        question (str): The question to be answered based on the form image.

    Returns:
        dict: The model's answer to the given question.

    Raises:
        FileNotFoundError: If the image_path does not lead to a valid file.
    """
    # Load the model
    model = LayoutLMv3ForQuestionAnswering.from_pretrained('hf-tiny-model-private/tiny-random-LayoutLMv3ForQuestionAnswering')
    tokenizer = LayoutLMv3Tokenizer.from_pretrained('hf-tiny-model-private/tiny-random-LayoutLMv3ForQuestionAnswering')

    # Load and preprocess the enrollment form image
    try:
        form_image = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'The image file was not found at the path: {image_path}')

    # Tokenize the text
    inputs = tokenizer(question, form_image, return_tensors='pt')

    # Ask questions and get the answer
    outputs = model(**inputs)
    return outputs

# test_function_code --------------------

def test_answer_questions_from_form_image():
    print("Testing started.")
    image_path = 'path/to/test/image.png'
    questions = [
        'What is the student\'s name?',
        'What is the student\'s age?',
        'What is the student\'s address?'
    ]
    expected_answers = ['John Doe', '12', '1234 Main St']

    for i, question in enumerate(questions):
        print(f"Testing case [{i + 1}/{len(questions)}] started.")
        answer = answer_questions_from_form_image(image_path, question)
        assert answer == expected_answers[i], f"Test case [{i + 1}/{len(questions)}] failed: Expected {expected_answers[i]}, got {answer}"
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_questions_from_form_image()