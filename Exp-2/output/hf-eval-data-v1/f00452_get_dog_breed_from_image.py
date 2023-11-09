from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
import requests

def get_dog_breed_from_image(img_url: str, question: str) -> str:
    '''
    This function takes in an image URL and a question as input, and returns the answer to the question.
    The question should be related to the image of the pet dogs.
    The function uses the Hugging Face Transformers library and the 'Salesforce/blip2-opt-2.7b' model to process the image and generate the answer.
    '''
    # Load the image from the provided URL
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    # Load the processor and the model
    processor = BlipProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
    model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
    # Process the image and the question
    inputs = processor(raw_image, question, return_tensors='pt')
    # Generate the output
    out = model.generate(**inputs)
    # Decode the output to get the answer
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer