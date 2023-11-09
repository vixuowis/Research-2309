from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
import requests

def get_food_ingredients(img_url: str, question: str = 'What are the ingredients of this dish?') -> str:
    '''
    This function takes an image URL and a question as input, processes the image using the BlipProcessor,
    generates a textual output using the Blip2ForConditionalGeneration model, and decodes the output to get a human-readable response.
    
    Parameters:
    img_url (str): The URL of the food image.
    question (str): The question to be answered by the model. Default is 'What are the ingredients of this dish?'.
    
    Returns:
    str: The decoded output from the model, providing information about the food item.
    '''
    processor = BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
    
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    inputs = processor(raw_image, question, return_tensors='pt')
    out = model.generate(**inputs)
    
    ingredient_info = processor.decode(out[0], skip_special_tokens=True)
    return ingredient_info