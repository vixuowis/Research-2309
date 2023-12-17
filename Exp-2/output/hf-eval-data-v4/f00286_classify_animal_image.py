# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_animal_image(image_path):
    """
    Classifies the image to determine whether it contains a cat, dog, or bird.

    Args:
    image_path (str): The path to the image file to be classified.

    Returns:
    dict: The classification result with class labels and confidence scores.
    """
    model = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
    class_names = ['cat', 'dog', 'bird']
    results = model(image_path, class_names=class_names)
    return results

# test_function_code --------------------

def test_classify_animal_image():
    print("Testing started.")
    # Test with an example image of a cat
    print("Testing image with a cat.")
    cat_result = classify_animal_image('cat_example.jpg')
    assert cat_result[0]['label'] == 'cat', f"Test failed, expected label 'cat' but got {cat_result[0]['label']}"
    print("Test passed for image with a cat.")

    # Test with an example image of a dog
    print("Testing image with a dog.")
    dog_result = classify_animal_image('dog_example.jpg')
    assert dog_result[0]['label'] == 'dog', f"Test failed, expected label 'dog' but got {dog_result[0]['label']}"
    print("Test passed for image with a dog.")

    print("Testing finished.")

# Run the test function
test_classify_animal_image()