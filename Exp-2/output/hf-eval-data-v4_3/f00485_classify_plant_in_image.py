# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_plant_in_image(image_path, possible_plant_names):
    """Classify the type of plant in the given image.

    Args:
        image_path (str): The file path to the image to classify.
        possible_plant_names (List[str]): A list of possible plant names for classification.

    Returns:
        str: The probable plant name for the given image.

    Raises:
        FileNotFoundError: If the image_path does not exist.
        ValueError: If possible_plant_names is empty.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    if not possible_plant_names:
        raise ValueError("List of possible plant names is empty.")

    clip_model = pipeline('image-classification', model='laion/CLIP-convnext_base_w-laion2B-s13B-b82K')
    plant_classifications = clip_model(image_path, possible_plant_names)
    top_plant = plant_classifications[0]['label']
    return top_plant

# test_function_code --------------------

def test_classify_plant_in_image():
    print("Testing started.")

    # Assume path to images and possible_names are correctly defined
    image_path = 'tests/resources/sample_plant.jpg'
    possible_names = ['rose', 'tulip', 'sunflower']

    # Testing case [1/3] started
    print("Testing case [1/3] started.")
    try:
        result = classify_plant_in_image(image_path, possible_names)
        assert result in possible_names, f"Test case [1/3] failed: Unexpected result {result}"
    except Exception as e:
        assert False, f"Test case [1/3] failed with an exception: {e}"

    # Testing case [2/3]: Test for non-existent image path
    print("Testing case [2/3] started.")
    try:
        classify_plant_in_image('non_existent.jpg', possible_names)
        assert False, "Test case [2/3] failed: FileNotFoundError not raised for non-existent image path"
    except FileNotFoundError:
        pass  # Expected
    except Exception as e:
        assert False, f"Test case [2/3] failed with an unexpected exception: {e}"

    # Testing case [3/3]: Test for empty possible names
    print("Testing case [3/3] started.")
    try:
        classify_plant_in_image(image_path, [])
        assert False, "Test case [3/3] failed: ValueError not raised for empty possible plant names"
    except ValueError:
        pass  # Expected
    except Exception as e:
        assert False, f"Test case [3/3] failed with an unexpected exception: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_classify_plant_in_image()