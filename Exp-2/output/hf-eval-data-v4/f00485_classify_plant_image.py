# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_plant_image(image_path, plant_names):
    """
    Classify the type of plant in an image using a pre-trained CLIP model.

    Args:
        image_path (str): The path to the image file to classify.
        plant_names (list): A list of possible plant names to classify against.

    Returns:
        str: The most probable plant name for the image.
    """
    # Create an image classification model
    clip = pipeline('image-classification', model='laion/CLIP-convnext_base_w-laion2B-s13B-b82K')

    # Classify the plant image with the provided plant names
    plant_classifications = clip(image_path, plant_names)

    # Get the top classification result
    top_plant = plant_classifications[0]['label']
    return top_plant

# test_function_code --------------------

def test_classify_plant_image():
    print("Testing classify_plant_image function.")

    # Here you would typically load a sample image from a dataset or use a placeholder
    sample_image_path = 'path/to/test_plant_image.jpg'
    # Define test plant names
    test_plant_names = ['rose', 'tulip', 'sunflower']

    # Expected result (as an example, actual result may vary)
    expected_result = 'rose'

    # Perform the classification
    result = classify_plant_image(sample_image_path, test_plant_names)

    # Check if the result is as expected
    assert result == expected_result, f'Test failed: Expected {expected_result}, got {result}'
    print("Test passed!")