# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_animal_image(image_path, categories):
    '''
    Classify an image of an animal into one of the categories provided.

    :param image_path: Path to the image file
    :param categories: List of class names (categories)
    :return: Dictionary containing classified categories with confidence scores
    '''
    # Load the pre-trained zero-shot image classification model
    classifier = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft')

    # Classify the image
    return classifier(image_path, categories)

# test_function_code --------------------

def test_classify_animal_image():
    print("Testing classify_animal_image function.")
    # Test with a sample image path and predefined categories
    image_path = 'sample_image.jpg'
    categories = ['cat', 'dog', 'bird', 'fish']

    # Expected outcome: The function should provide a dictionary with confidence scores for each category
    result = classify_animal_image(image_path, categories)
    assert isinstance(result, dict), "The result should be a dictionary with scores for each category."
    assert all(isinstance(score, float) for score in result.values()), "All confidence scores should be floating-point numbers."

    print("Testing finished successfully.")

# Run the test function
test_classify_animal_image()