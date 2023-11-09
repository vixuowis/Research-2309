# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_plant(image_path: str, labels: list) -> str:
    """
    Classify the type of plant in the image using a pretrained model.

    Args:
        image_path (str): The path to the image file.
        labels (list): A list of possible class names.

    Returns:
        str: The most probable plant name.
    """
    clip = pipeline('image-classification', model='laion/CLIP-convnext_base_w-laion2B-s13B-b82K')
    plant_classifications = clip(image_path, labels)
    top_plant = plant_classifications[0]['label']
    return top_plant

# test_function_code --------------------

def test_classify_plant():
    """
    Test the classify_plant function.
    """
    image_path = 'path/to/test_image.jpg'
    labels = ['rose', 'tulip', 'sunflower']
    result = classify_plant(image_path, labels)
    assert result in labels, f'Error: {result} not in {labels}'

# call_test_function_code --------------------

test_classify_plant()