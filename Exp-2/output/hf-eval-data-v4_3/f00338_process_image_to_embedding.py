# requirements_file --------------------

import subprocess

requirements = ["vc_models", "PIL"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from vc_models.models.vit import model_utils

# function_code --------------------

def process_image_to_embedding(image_path):
    """
    Processes an image and returns the embedding using the pre-trained VC-1 model.
    
    Args:
        image_path (str): The file path to the image to be processed.
    Returns:
        ndarray: The embedding obtained from the VC-1 model.
    Raises:
        IOError: If the image_path does not exist or is not accessible.
    """
    # Load the VC-1 model and associated utilities
    model, embd_size, model_transforms, _ = model_utils.load_model(model_utils.VC1_BASE_NAME)

    # Load and preprocess the image
    img = Image.open(image_path)
    transformed_img = model_transforms(img)
    
    # Obtain embedding from the model
    embedding = model(transformed_img)

    return embedding

# test_function_code --------------------

def test_process_image_to_embedding():
    print("Testing started.")
    
    # Test case 1: Check if embedding is returned for a valid image path
    valid_image_path = 'path/to/valid/image.jpg'
    print("Testing case [1/3] started.")
    valid_embedding = process_image_to_embedding(valid_image_path)
    assert valid_embedding is not None, f"Test case [1/3] failed: Expected an embedding, got None"

    # Test case 2: Check that IOError is raised for an invalid image path
    invalid_image_path = 'path/to/nonexistent/image.jpg'
    print("Testing case [2/3] started.")
    try:
        process_image_to_embedding(invalid_image_path)
        assert False, "Test case [2/3] failed: Expected IOError for invalid image path"
    except IOError:
        assert True

    print("Testing finished.")

# call_test_function_line --------------------

test_process_image_to_embedding()