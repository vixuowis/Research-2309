from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import requests

# Function to detect intruder using multimodal visual question answering
# @param: cctv_image_path - Path to the CCTV image
# @return: Answer to the question 'Who entered the room?'
def detect_intruder(cctv_image_path):
    # Load the pretrained model
    processor = BlipProcessor.from_pretrained('Salesforce/blip-vqa-capfilt-large')
    model = BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-capfilt-large')

    # Open the image and convert it to RGB
    cctv_image = Image.open(cctv_image_path)

    # Define the question
    question = 'Who entered the room?'

    # Process the image and question
    inputs = processor(cctv_image, question, return_tensors='pt')

    # Generate the answer
    answer = model.generate(**inputs)

    # Decode the answer and return it
    return processor.decode(answer[0], skip_special_tokens=True)