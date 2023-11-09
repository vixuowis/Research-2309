from transformers import pipeline


def generate_image_description(image):
    """
    This function generates a description for an image using the 'microsoft/git-large-r-textcaps' model from Hugging Face Transformers.
    The model has been fine-tuned on the TextCaps dataset and is capable of generating image descriptions based on the content of the image.
    
    Parameters:
    image (str): Path to the image file
    
    Returns:
    str: Generated description of the image
    """
    # Create a text generation model using the pipeline function
    description_generator = pipeline('text-generation', model='microsoft/git-large-r-textcaps')
    
    # Generate a description for the given input image
    image_description = description_generator(image)
    
    return image_description