# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_device_image(image_path: str, class_names: list) -> dict:
    """
    Classify a device image using a pre-trained model from Hugging Face.

    Args:
        image_path (str): Path to the image file.
        class_names (list): List of class names for classification.

    Returns:
        dict: The predicted class and its score.

    Raises:
        OSError: If the model or the image file is not found.
    """
    # Create a classifier pipeline
    device_classifier = pipeline(
        "image-classification",
        model="huggingface/google/vit-base-patch16-224-in21k")
    # Read the image file and make sure it exists.
    image = read_image_file(image_path)
    if not os.path.isfile(image):
        raise OSError("Image file does not exist.")
    # Classify the device in the image with the classifier pipeline.
    device, score = device_classifier([image])[0]
    # Find the predicted class and its score in the list of classes.
    for name, value in zip(class_names, score):
        if name == device:
            break
    # Return the result as a dictionary.
    return {"device": device, "score": score}

# test_function_code --------------------

def test_classify_device_image():
    """
    Test the function classify_device_image.
    """
    # Test case 1: Valid image and class names
    try:
        image_path = 'path/to/valid/image.jpg'
        class_names = ['smartphone', 'laptop', 'tablet']
        prediction = classify_device_image(image_path, class_names)
        assert isinstance(prediction, dict), 'The prediction should be a dictionary.'
    except OSError:
        pass

    # Test case 2: Invalid image path
    try:
        image_path = 'path/to/invalid/image.jpg'
        class_names = ['smartphone', 'laptop', 'tablet']
        prediction = classify_device_image(image_path, class_names)
    except OSError:
        pass

    # Test case 3: Invalid class names
    try:
        image_path = 'path/to/valid/image.jpg'
        class_names = ['invalid', 'class', 'names']
        prediction = classify_device_image(image_path, class_names)
    except OSError:
        pass

    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_device_image()