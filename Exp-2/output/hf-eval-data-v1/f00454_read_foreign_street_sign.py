from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
from PIL import Image
import requests

# Function to read street signs in a foreign language
# Uses the MgpstrForSceneTextRecognition model from Hugging Face Transformers
# The model is trained on MJSynth and SynthText datasets
# It is a pure vision Scene Text Recognition (STR) model, consisting of ViT and specially designed A^3 modules
# Can be used for optical character recognition (OCR) on text images

def read_foreign_street_sign(image_url):
    # Instantiate the MgpstrProcessor and MgpstrForSceneTextRecognition using the 'alibaba-damo/mgp-str-base' model
    processor = MgpstrProcessor.from_pretrained('alibaba-damo/mgp-str-base')
    model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base')

    # Convert the image of the street sign into a format that can be fed into the model as input
    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    pixel_values = processor(images=image, return_tensors='pt').pixel_values

    # Use the model to recognize the text from the street sign image
    outputs = model(pixel_values)

    # Decode the text and return it
    generated_text = processor.batch_decode(outputs.logits)['generated_text']
    return generated_text