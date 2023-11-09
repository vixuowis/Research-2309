from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration
import requests

def identify_landmark(img_url: str, question: str = 'What is the name of this landmark?') -> str:
    '''
    Function to identify landmarks using the BLIP-2 model.
    Args:
    img_url : str : URL of the image of the landmark
    question : str : Question to be answered by the model based on the image. Default is 'What is the name of this landmark?'
    Returns:
    str : Answer or information about the landmark
    '''
    processor = BlipProcessor.from_pretrained('Salesforce/blip2-flan-t5-xl')
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-flan-t5-xl')
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    inputs = processor(raw_image, question, return_tensors='pt')
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer