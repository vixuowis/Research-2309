# function_import --------------------

from transformers import pipeline
import PIL.Image

# function_code --------------------

def classify_image(image_path):
    """
    Classify the object in the image using a pre-trained model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: The classification result, which includes the possible object categories and their probabilities.
    """
    image_classifier = pipeline('image-classification', model='timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k', framework='pt')
    image = PIL.Image.open(image_path)
    result = image_classifier(image)
    return result

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function.

    Raises:
        AssertionError: If the function does not work as expected.
    """
    test_image_path = 'path/to/test/image.jpg'
    result = classify_image(test_image_path)
    assert isinstance(result, list), 'The result should be a list.'
    assert 'label' in result[0], 'Each item in the result should have a label.'
    assert 'score' in result[0], 'Each item in the result should have a score.'

# call_test_function_code --------------------

test_classify_image()