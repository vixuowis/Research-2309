from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Function to generate descriptive captions for photographs related to the products
# using the Hugging Face Transformers library and the Salesforce/blip-image-captioning-base model

def generate_image_caption(image_path, text='product photography'):
    '''
    Function to generate descriptive captions for photographs related to the products
    Args:
    image_path : str : path to the image
    text : str : context to the photograph, default is 'product photography'
    Returns:
    str : generated caption for the image
    '''
    # Initialize the processor and the model
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
    
    # Open the image
    image = Image.open(image_path)
    
    # Process the image and text
    inputs = processor(image, text, return_tensors='pt')
    
    # Generate the caption
    out = model.generate(**inputs)
    
    # Decode and return the caption
    return processor.decode(out[0], skip_special_tokens=True)