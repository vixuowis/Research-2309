# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_pet_images(image_path, model_name='laion/CLIP-convnext_base_w-laion2B-s13B-b82K'):
    # Initialize the zero-shot image classification pipeline with the specified model
    clip = pipeline('image-classification', model=model_name)

    # Define the class labels for pets
    pet_labels = ['cat', 'dog']

    # Classify the image and return the results
    classification_result = clip(image_path, pet_labels)
    return classification_result

# test_function_code --------------------

def test_classify_pet_images():
    print("Testing classify_pet_images function.")

    # Test case 1: Classify an example cat image
    cat_image_path = 'path/to/example_cat.jpg'
    cat_result = classify_pet_images(cat_image_path)
    assert 'cat' in cat_result[0]['label'], f"Test case failed: Expected label 'cat' not found in result {cat_result}"

    # Test case 2: Classify an example dog image
    dog_image_path = 'path/to/example_dog.jpg'
    dog_result = classify_pet_images(dog_image_path)
    assert 'dog' in dog_result[0]['label'], f"Test case failed: Expected label 'dog' not found in result {dog_result}"

    print("All tests passed!")

# Run the test function
test_classify_pet_images()