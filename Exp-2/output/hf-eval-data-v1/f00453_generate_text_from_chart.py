from transformers import Pix2StructForConditionalGeneration
import PIL.Image

# Function to generate text from chart using Pix2Struct model
# @param image_path: The path to the image file
# @return: The generated text describing the chart

def generate_text_from_chart(image_path):
    # Load the pre-trained model
    model = Pix2StructForConditionalGeneration.from_pretrained('google/pix2struct-chartqa-base')
    # Open the image file
    image = PIL.Image.open(image_path)
    # Generate text from the image
    generated_text = model.generate_text(image)
    return generated_text