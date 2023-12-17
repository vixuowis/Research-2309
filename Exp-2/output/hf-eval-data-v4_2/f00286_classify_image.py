# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_image(image_path):
    """
    Classify an image to determine whether it contains a cat, dog, or bird.

    Args:
        image_path (str): The path to the image file to be classified.

    Returns:
        dict: A dictionary with the predicted category and confidence score.

    Raises:
        FileNotFoundError: If the image file does not exist at the given path.
        ValueError: If the image cannot be processed by the model.
    """
    model = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
    class_names = ['cat', 'dog', 'bird']
    try:
        results = model(image_path, class_names=class_names)
        return results
    except FileNotFoundError:
        raise FileNotFoundError(f'The image file was not found at the given path: {image_path}')
    except Exception as e:
        raise ValueError(f'An error occurred while processing the image: {e}')

# test_function_code --------------------

def test_classify_image():
    print('Testing started.')
    # Assuming 'sample_images' contains paths to cat, dog, bird images for testing
    test_images = ['sample_images/cat.jpg', 'sample_images/dog.jpg', 'sample_images/bird.jpg']

    for index, image_path in enumerate(test_images):
        case_number = index + 1
        total_cases = len(test_images)
        print(f'Testing case [{case_number}/{total_cases}] started.')
        try:
            results = classify_image(image_path)
            assert any(result['label'] in image_path for result in results), f'Test case [{case_number}/{total_cases}] failed: Incorrect classification.'
        except Exception as e:
            assert False, f'Test case [{case_number}/{total_cases}] failed with an exception: {e}'

    print('Testing finished.')

# call_test_function_line --------------------

test_classify_image()