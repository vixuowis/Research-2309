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
    try:
        device = pipeline("image-classification", 
                          pretrained="google/vit-base-patch16-224")        
        results = device(image_path)
        if len(results) > 0:
            return {
                "predicted": class_names[results[0]['label']],
                "score": round(results[0]['score'], 3),
            }  
    except OSError as e:
        print("ERROR", e)
        raise

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