# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline
import PIL.Image

# function_code --------------------

def classify_image(image_path):
    """
    Classify an image using a pre-trained model from Hugging Face Transformers.

    Args:
        image_path (str): The file path to the image to be classified.

    Returns:
        list: A list of dictionaries containing the possible object categories and their probabilities.
    """
    image_classifier = pipeline('image-classification', model='timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k', framework='pt')
    image = PIL.Image.open(image_path)
    result = image_classifier(image)
    return result

# test_function_code --------------------

def test_classify_image():
    print("Testing the 'classify_image' function.")

    # Test with a sample image
    sample_image_path = 'sample_image.jpg'  # A sample image file path
    classification_result = classify_image(sample_image_path)

    # Check if the result is a list of predicted categories
    assert type(classification_result) is list, "The classification result should be a list."

    # Check if each item in the list is a dictionary with 'label' and 'score' keys
    for item in classification_result:
        assert 'score' in item and 'label' in item, "Each item in the classification result should contain 'label' and 'score'."

    print("All tests passed for 'classify_image'.")

# Run the test
print("Running classify_image function tests:")
test_classify_image()