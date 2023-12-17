# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import LayoutLMv3ForQuestionAnswering, LayoutLMv3Tokenizer
import torch

# function_code --------------------

def extract_information(image_path: str, questions: list) -> dict:
    """
    Extracts answers to specific questions from a scanned document image.

    Args:
        image_path (str): The path to the image file of the scanned document.
        questions (list): A list of questions to answer based on the document.

    Returns:
        dict: A dictionary mapping each question to its extracted answer.

    Raises:
        ValueError: If image_path is not valid or questions list is empty.
    """
    tokenizer = LayoutLMv3Tokenizer.from_pretrained('hf-tiny-model-private/tiny-random-LayoutLMv3ForQuestionAnswering')
    model = LayoutLMv3ForQuestionAnswering.from_pretrained('hf-tiny-model-private/tiny-random-LayoutLMv3ForQuestionAnswering')

    if not questions:
        raise ValueError('Questions list cannot be empty.')

    answers = {}
    for question in questions:
        input_data = tokenizer(question, image_path, return_tensors="pt")
        output = model(**input_data)
        answer_start, answer_end = output.start_logits.argmax(), output.end_logits.argmax() + 1
        answer_tokens = tokenizer.convert_ids_to_tokens(answer_start, answer_end)
        answers[question] = ' '.join(answer_tokens)

    return answers

# test_function_code --------------------

def test_extract_information():
    print("Testing started.")
    # Load a sample document image and define test questions
    image_path = 'path/to/test/image.jpg'
    test_questions = ['What is the total amount?', 'When is the due date?']

    # Test case 1: Valid case with two questions
    print("Testing case [1/2] started.")
    try:
        answers = extract_information(image_path, test_questions)
        assert isinstance(answers, dict) and len(answers) == 2, "Test case [1/2] failed: The number of answers does not match the number of questions."
    except Exception as e:
        assert False, f"Test case [1/2] failed: {e}"

    # Test case 2: Error case no questions
    print("Testing case [2/2] started.")
    try:
        extract_information(image_path, [])
        assert False, "Test case [2/2] failed: Exception for empty question list not raised."
    except ValueError as e:
        assert str(e) == 'Questions list cannot be empty.', "Test case [2/2] failed: Wrong error message."

    print("Testing finished.")

# call_test_function_line --------------------

test_extract_information()