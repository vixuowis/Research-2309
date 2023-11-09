from PIL import Image
import requests
from transformers import BlipProcessor, Blip2ForConditionalGeneration

def get_artwork_info(image_path: str, question: str) -> str:
    '''
    This function takes in the path of an artwork image and a question about the artwork.
    It uses the Blip2ForConditionalGeneration model from Hugging Face Transformers to generate an answer to the question based on the image.
    
    Parameters:
    image_path (str): The path to the artwork image.
    question (str): The question about the artwork.
    
    Returns:
    str: The answer to the question.
    '''
    # Initialize the processor and model objects by loading the pre-trained model 'Salesforce/blip2-flan-t5-xl'.
    processor = BlipProcessor.from_pretrained('Salesforce/blip2-flan-t5-xl')
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-flan-t5-xl')
    
    # Load the image of the artwork using the Image module and convert it to RGB format.
    raw_image = Image.open(image_path).convert('RGB')
    
    # Pass the image and the question to the processor, which will process and return the necessary tensors.
    inputs = processor(raw_image, question, return_tensors='pt')
    
    # Use the model to generate a response based on the processed input tensors.
    out = model.generate(**inputs)
    
    # Decode the output to get the answer to your question.
    answer = processor.decode(out[0], skip_special_tokens=True)
    
    return answer