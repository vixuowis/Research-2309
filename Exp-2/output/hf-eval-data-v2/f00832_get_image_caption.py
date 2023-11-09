# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_image_caption(image_path: str, question: str) -> str:
    """
    This function uses a pre-trained model to perform visual question answering tasks in the Polish language.
    It takes an image and a question as input and returns an answer to the question based on the image.

    Args:
        image_path (str): The path to the image.
        question (str): The question to be answered.

    Returns:
        str: The answer to the question based on the image.
    """
    vqa_pipeline = pipeline('visual-question-answering', model='azwierzc/vilt-b32-finetuned-vqa-pl')
    answer = vqa_pipeline(image_path, question)
    return answer

# test_function_code --------------------

def test_get_image_caption():
    """
    This function tests the 'get_image_caption' function.
    It uses a sample image and a question to test the function.
    """
    image_path = 'path_to_test_image.jpg'
    question = 'Jakie są główne kolory na zdjęciu?'
    answer = get_image_caption(image_path, question)
    assert isinstance(answer, str), 'The function should return a string.'

# call_test_function_code --------------------

test_get_image_caption()