# requirements_file --------------------

!pip install -U pillow, numpy, torch, transformers

# function_import --------------------

from vc_models.models.vit import model_utils

# function_code --------------------

def extract_features_from_image(image):
    """
    This function takes an image as input, processes it using the
    VC-1 model's preprocessing utilities, then passes the image through
    the model to extract features for use in EmbodiedAI tasks.
    
    :param image: A PIL.Image or a tensor representing the input image.
    :return: An embedding representing the visual information from the image.
    """
    # Load the pre-trained VC-1 Model
    model, embd_size, model_transforms, model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)

    # Transform the image using the model's preprocessing utilities
    transformed_img = model_transforms(image)

    # Extract features by passing the transformed image through the model
    embedding = model(transformed_img)
    
    return embedding

# test_function_code --------------------

def test_extract_features_from_image():
    print("Testing started.")
    # Assuming a function 'load_sample_image' exists to load images for the test
    image = load_sample_image()
    
    # Testing extraction of features
    print("Testing feature extraction function.")
    embedding = extract_features_from_image(image)

    # Replace the ... below with relevant assertions for the expected output
    assert embedding is not None, f"Test failed: The embedding should not be None"
    assert embedding.shape[0] == embd_size, f"Test failed: Embedding size mismatch"

    # Additional test cases can be added to validate other aspects of the feature extraction
    # For example, ensure that the embedding values are within the expected range
    
    print("Testing finished.")

# Function to load a sample image for the tests (placeholder)
def load_sample_image():
    # Here we should load an actual image from a dataset
    # For the purpose of this example, we just return a placeholder
    from PIL import Image
    import numpy as np
    return Image.fromarray(np.uint8(np.random.rand(224,224,3) * 255))

test_extract_features_from_image()