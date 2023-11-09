# function_import --------------------

from vc_models.models.vit import model_utils

# function_code --------------------

def get_image_embedding(img):
    """
    This function captures the elderly's activities as images, transforms the image data into a format that the model can understand, and then passes the transformed image through the model to obtain an embedding.

    Args:
        img (PIL.Image): The image captured by the robot's camera.

    Returns:
        torch.Tensor: The embedding of the image.
    """
    model, embd_size, model_transforms, model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
    transformed_img = model_transforms(img)
    embedding = model(transformed_img)
    return embedding

# test_function_code --------------------

def test_get_image_embedding():
    """
    This function tests the get_image_embedding function by loading a sample image and checking the shape of the returned embedding.
    """
    img = Image.open('sample.jpg')
    embedding = get_image_embedding(img)
    assert embedding.shape[0] == embd_size

# call_test_function_code --------------------

test_get_image_embedding()