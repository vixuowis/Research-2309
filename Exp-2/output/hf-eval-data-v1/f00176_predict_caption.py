from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# Load the pre-trained model from Hugging Face Transformers
model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning')

# Create an instance of the ViTImageProcessor and AutoTokenizer classes
feature_extractor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
tokenizer = AutoTokenizer.from_pretrained('nlpconnect/vit-gpt2-image-captioning')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to predict caption for an image
# Input: image_path - Path to the image
# Output: Caption for the image

def predict_caption(image_path):
    # Open the image
    input_image = Image.open(image_path)
    # Convert the image to RGB if it is not
    if input_image.mode != "RGB":
        input_image = input_image.convert(mode="RGB")

    # Extract features from the image
    pixel_values = feature_extractor(images=[input_image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    # Generate caption for the image
    output_ids = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption.strip()