# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_food_image(image_path: str, question: str) -> dict:
    """
    This function uses a Visual Question Answering model to analyze an image in relation to food and answer a question about it.

    Args:
        image_path (str): The path to the image to be analyzed.
        question (str): The question to be answered about the image.

    Returns:
        dict: The answer to the question about the image.
    """
    vqa_model = pipeline('visual-question-answering', model='azwierzc/vilt-b32-finetuned-vqa-pl')
    answer = vqa_model({'image': image_path, 'question': question})
    return answer

# test_function_code --------------------

def test_analyze_food_image():
    """
    This function tests the analyze_food_image function.
    """
    image_path = 'path_to_test_image'
    question = 'Jakie składniki są w daniu?'
    answer = analyze_food_image(image_path, question)
    assert isinstance(answer, dict), 'The function should return a dictionary.'
    assert 'answer' in answer, 'The dictionary should contain an answer.'

# call_test_function_code --------------------

test_analyze_food_image()