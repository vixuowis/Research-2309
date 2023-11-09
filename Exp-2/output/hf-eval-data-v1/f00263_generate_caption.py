from transformers import pipeline


def generate_caption(image_path):
    """
    This function generates a caption for an image using the 'salesforce/blip2-opt-6.7b' model from Hugging Face Transformers.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    str: The generated caption for the image.
    """
    # Load the model
    caption_generator = pipeline('text2text-generation', model='salesforce/blip2-opt-6.7b')
    
    # Generate the caption
    caption = caption_generator(image_path)
    
    return caption