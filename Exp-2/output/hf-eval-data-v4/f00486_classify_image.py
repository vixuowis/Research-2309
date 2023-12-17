# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_image(image_path):
    # Load the pre-trained image classification model from Hugging Face
    classifier = pipeline('image-classification', model='laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg')

    # Define the class names for classification
    class_names = ['landscape', 'cityscape', 'beach', 'forest', 'animals']

    # Run the classifier on the image
    results = classifier(image_path, class_names=class_names)

    # Return the classification result
    return results

# test_function_code --------------------

def test_classify_image():
    print("Testing classify_image function")

    # Define a test image path
    test_image_path = 'test_image.jpg'  # Replace with a valid image path

    # Perform image classification
    classification_results = classify_image(test_image_path)

    # Verify that the results is a list of predictions
    assert isinstance(classification_results, list), "The classification result should be a list"

    # Verify that each item in the list is a dictionary with the required keys
    for result in classification_results:
        assert set(result.keys()) == {'score', 'label'}, "Each item in the result list should be a dictionary with 'score' and 'label'"

    print("Test passed")

# Run the test
if __name__ == '__main__':
    test_classify_image()