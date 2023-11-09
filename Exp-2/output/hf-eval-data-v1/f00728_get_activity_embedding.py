from vc_models.models.vit import model_utils


def get_activity_embedding(img):
    """
    This function captures the elderly's activities as images, transforms the image data into a format that the model can understand, and passes the transformed image through the model to obtain an embedding.
    The embedding can be used to understand the scene and make decisions on how the robot should respond to the elderly's current activities.
    
    Args:
        img (Image): The image captured by the robot's camera.
    
    Returns:
        Tensor: The embedding of the image.
    """
    # Load the pretrained model
    model, embd_size, model_transforms, model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
    
    # Transform the image
    transformed_img = model_transforms(img)
    
    # Get the embedding of the image
    embedding = model(transformed_img)
    
    return embedding