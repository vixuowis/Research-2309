# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_car_damage(image_path):
    """
    Classifies car damage as either 'major accident' or 'minor damages'.

    Args:
        image_path (str): The path to the image of the car that needs to be classified.

    Returns:
        dict: A dictionary containing the classification result and confidence score.

    Raises:
        FileNotFoundError: If the image_path does not exist.
        Exception: If the image classification pipeline throws an error.
    """
    # Define the classification categories
    class_names = ['major accident', 'minor damages']

    # Load the pre-trained CLIP model and create the classifier pipeline
    classifier = pipeline('image-classification', model='laion/CLIP-ViT-B-16-laion2B-s34B-b88K')

    # Perform classification on the given image
    result = classifier(image_path, class_names)
    return result

# test_function_code --------------------

def test_classify_car_damage():
    print("Testing started.")
    # Assume 'load_dataset' is available from a dataset library
    dataset = load_dataset("some_car_damage_dataset")
    sample_data = dataset[0]  # Sample image data from the dataset

    # Test case 1: Image shows a major accident
    print("Testing case [1/3] started.")
    result = classify_car_damage(sample_data['major_accident_image'])
    assert result[0]['label'] == 'major accident', f"Test case [1/3] failed: {result}"

    # Test case 2: Image shows minor damages
    print("Testing case [2/3] started.")
    result = classify_car_damage(sample_data['minor_damages_image'])
    assert result[0]['label'] == 'minor damages', f"Test case [2/3] failed: {result}"

    # Test case 3: Image path is invalid
    print("Testing case [3/3] started.")
    try:
        classify_car_damage('invalid_image_path')
        assert False, "Test case [3/3] failed: No exception raised for an invalid image path."
    except FileNotFoundError:
        pass  # Expected result
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_car_damage()