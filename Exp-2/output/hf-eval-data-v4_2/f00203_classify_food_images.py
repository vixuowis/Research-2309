# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_food_images(image_paths, class_names):
    """
    Classify a batch of images into predefined food classes using a pre-trained model.

    Args:
        image_paths (List[str]): List of paths for the images to be classified.
        class_names (List[str]): List of possible food class names.

    Returns:
        List[Dict[str, Any]]: List of dictionaries with classification results.

    Raises:
        FileNotFoundError: If any of the image paths does not exist.
        ValueError: If `class_names` is empty.
    """
    # Ensure class_names is not empty
    if not class_names:
        raise ValueError('Class names list cannot be empty.')

    # Initialize the classifier
    image_classifier = pipeline('image-classification', model='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')

    # Classify the images
    results = []
    for image_path in image_paths:
        # Check if image path exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'Image path {image_path} does not exist.')
        # Classify and append to results
        result = image_classifier(image_path, possible_class_names=class_names)
        results.append(result)

    return results

# test_function_code --------------------

def test_classify_food_images():
    print("Testing started.")
    # Assuming `load_dataset` function and sample dataset are available
    sample_images, sample_class_names = load_dataset('sample_food_images')
    sample_image_path = sample_images[0]
    # Test with a valid case
    print("Testing case [1/1] started.")
    try:
        results = classify_food_images([sample_image_path], sample_class_names)
        assert len(results) == 1 and isinstance(results, list), f"Test case [1/1] failed: Expected list with one result."
        print("Testing case [1/1] passed.")
    except Exception as e:
        print(f"Test case [1/1] failed: {str(e)}")
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_food_images()