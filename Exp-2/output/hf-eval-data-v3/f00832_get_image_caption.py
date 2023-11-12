# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_image_caption(image_path: str, question: str) -> str:
    '''
    This function uses a pre-trained model to perform visual question answering tasks in the Polish language.
    It takes an image path and a question as input, processes the image, and returns an answer to the question.

    Args:
        image_path (str): The path to the image file.
        question (str): The question related to the image.

    Returns:
        str: The answer to the question.

    Raises:
        ValueError: If the image_path is not a valid path to an image file or a valid URL starting with `http://` or `https://`, or if the image_path is not a base64 encoded string.
    '''
    vqa_pipeline = pipeline('visual-question-answering', model='azwierzc/vilt-b32-finetuned-vqa-pl')
    answer = vqa_pipeline(image_path, question)
    return answer

# test_function_code --------------------

def test_get_image_caption():
    '''
    This function tests the get_image_caption function with different test cases.
    '''
    # Test case 1: A valid image URL and a valid question
    image_path = 'https://placekitten.com/200/300'
    question = 'Jakie są główne kolory na zdjęciu?'
    answer = get_image_caption(image_path, question)
    assert isinstance(answer, str), 'The answer should be a string.'

    # Test case 2: A valid image URL and a different valid question
    image_path = 'https://placekitten.com/200/300'
    question = 'Czy na zdjęciu jest kot?'
    answer = get_image_caption(image_path, question)
    assert isinstance(answer, str), 'The answer should be a string.'

    # Test case 3: A different valid image URL and a valid question
    image_path = 'https://placekitten.com/300/400'
    question = 'Jakie są główne kolory na zdjęciu?'
    answer = get_image_caption(image_path, question)
    assert isinstance(answer, str), 'The answer should be a string.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_image_caption()