# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_image_question(image_path, question):
    # Initialize the visual question-answering pipeline with the 'JosephusCheung/GuanacoVQAOnConsumerHardware' model.
    vqa = pipeline('visual-question-answering', model='JosephusCheung/GuanacoVQAOnConsumerHardware')
    
    # Use the created pipeline to process the image and question text.
    answer = vqa(image_path, question)
    return answer

# test_function_code --------------------

def test_answer_image_question():
    print("Testing started.")
    image_path = 'test_image.jpg'
    question = 'What is depicted in the image?'
    expected_answer = 'test_answer'  # Place a presumed correct answer here based on the test image

    # Test case
    print("Testing the answer_image_question function.")
    result = answer_image_question(image_path, question)
    assert result == expected_answer, f"Test failed: {result} does not match the expected answer {expected_answer}"
    print("Test passed.")

test_answer_image_question()