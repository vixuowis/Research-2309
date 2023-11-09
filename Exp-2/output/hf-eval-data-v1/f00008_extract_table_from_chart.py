from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import requests
from PIL import Image

def extract_table_from_chart(chart_url):
    '''
    This function extracts a linearized table from a given chart image URL.
    It uses the Pix2StructForConditionalGeneration model from the Hugging Face Transformers library.
    
    Parameters:
    chart_url (str): The URL of the chart image.
    
    Returns:
    str: The extracted table as a linearized text.
    '''
    # Load the pre-trained model and processor
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    processor = Pix2StructProcessor.from_pretrained('google/deplot')
    
    # Open the image file that contains the chart
    image = Image.open(requests.get(chart_url, stream=True).raw)
    
    # Convert the image into the required format
    inputs = processor(images=image, text='Generate underlying data table of the figure below:', return_tensors='pt')
    
    # Process the image and generate the underlying data table as a linearized text
    predictions = model.generate(**inputs, max_new_tokens=512)
    table = processor.decode(predictions[0], skip_special_tokens=True)
    
    return table