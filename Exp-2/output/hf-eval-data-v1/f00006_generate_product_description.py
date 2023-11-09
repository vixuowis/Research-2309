from transformers import GenerativeImage2TextModel


def generate_product_description(product_image):
    """
    This function generates a product description for an image-based online store platform.
    It uses the GenerativeImage2TextModel from the transformers library provided by Hugging Face.
    The model is specifically trained for image-to-text transformation tasks, and is ideal for creating product descriptions.
    
    Args:
    product_image (str): The path to the product image.
    
    Returns:
    str: The generated product description.
    """
    # Load the model
    git_model = GenerativeImage2TextModel.from_pretrained('microsoft/git-large-coco')
    
    # Generate the product description
    product_description = git_model.generate_image_description(product_image)
    
    return product_description