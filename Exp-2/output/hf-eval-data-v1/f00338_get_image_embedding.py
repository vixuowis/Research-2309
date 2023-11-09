from vc_models.models.vit import model_utils


def get_image_embedding(img):
    """
    This function takes an image as input and returns an embedding that represents the visual information from the image.
    The embedding is obtained by passing the image through a pre-trained VC-1 model.
    The VC-1 model is a vision transformer (ViT) pre-trained on over 4,000 hours of egocentric videos from 7 different sources, together with ImageNet.
    The model is intended for use for EmbodiedAI tasks, such as object manipulation and indoor navigation.
    
    Parameters:
    img (Image): The input image.
    
    Returns:
    Tensor: The embedding that represents the visual information from the image.
    """
    # Load the pre-trained VC-1 model
    model, embd_size, model_transforms, model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
    
    # Preprocess the image using the transformation function provided by the model
    transformed_img = model_transforms(img)
    
    # Pass the transformed image through the loaded VC-1 model to obtain an embedding
    embedding = model(transformed_img)
    
    return embedding