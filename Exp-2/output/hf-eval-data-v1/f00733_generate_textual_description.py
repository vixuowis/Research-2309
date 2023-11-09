from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import torch

# Function to generate textual descriptions for images
# Uses the Pix2StructForConditionalGeneration model from Hugging Face Transformers
# The model is pretrained on image-text pairs for various tasks, including image captioning and visual question answering
# It can achieve state-of-the-art results in six out of nine tasks across four domains: documents, illustrations, user interfaces, and natural images

def generate_textual_description(image_path):
    # Load the pre-trained Pix2Struct model
    model = Pix2StructForConditionalGeneration.from_pretrained('google/pix2struct-base')
    # Initialize the Pix2StructProcessor to process the input images
    processor = Pix2StructProcessor.from_pretrained('google/pix2struct-base')
    # Process the input images before feeding them to the model
    inputs = processor(images=[image_path], return_tensors="pt")
    # Generate textual descriptions for the input images
    outputs = model.generate(**inputs)
    # Decode the generated text
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return generated_text