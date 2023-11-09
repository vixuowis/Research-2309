from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import requests
from PIL import Image


def analyze_chart_image(url):
    '''
    This function takes a URL of a chart image as input and returns a summary of the information contained within the chart.
    It uses the Pix2StructForConditionalGeneration model from the Hugging Face Transformers library.
    '''
    # Load the 'google/deplot' model and processor
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    processor = Pix2StructProcessor.from_pretrained('google/deplot')

    # Open the chart image using the PIL library and the image URL
    image = Image.open(requests.get(url, stream=True).raw)

    # Prepare the inputs for the model
    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")

    # Generate a summary of the image
    predictions = model.generate(**inputs, max_new_tokens=512)

    # Decode the summary
    summary = processor.decode(predictions[0], skip_special_tokens=True)

    return summary