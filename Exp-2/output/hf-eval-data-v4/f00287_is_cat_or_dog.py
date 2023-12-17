# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def is_cat_or_dog(image_path):
    # Load the pre-trained image classification pipeline
    image_classifier = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft')

    # Apply the classifier to the image with 'cat' and 'dog' as the categories
    result = image_classifier(image_path, ['cat', 'dog'])

    # Return the classification result
    return result

# test_function_code --------------------

def test_is_cat_or_dog():
    print("Testing started.")

    # Test case 1: A known cat image
    cat_result = is_cat_or_dog('path/to/cat_image.jpg')
    print("Testing case [1/2] started.")
    assert cat_result[0]['label'] == 'cat', f"Test case [1/2] failed: Expected 'cat', got {cat_result[0]['label']}"

    # Test case 2: A known dog image
    dog_result = is_cat_or_dog('path/to/dog_image.jpg')
    print("Testing case [2/2] started.")
    assert dog_result[0]['label'] == 'dog', f"Test case [2/2] failed: Expected 'dog', got {dog_result[0]['label']}"
    print("Testing finished.")

# Run the test function
test_is_cat_or_dog()