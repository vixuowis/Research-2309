# requirements_file --------------------

!pip install -U transformers torch PIL

# function_import --------------------

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# function_code --------------------

def generate_image_captions(image_paths):
    # Load the pre-trained vision-language model
    model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
    feature_extractor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
    tokenizer = AutoTokenizer.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define generation keyword arguments
    gen_kwargs = {"max_length": 16, "num_beams": 4}

    # Process images and generate captions
    captions = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        pixel_values = feature_extractor(images=[image], return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(device)
        output_ids = model.generate(pixel_values, **gen_kwargs)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        captions.append(caption)

    return captions

# test_function_code --------------------

def test_generate_image_captions():
    print("Testing generate_image_captions function.")

    # You should replace this path with a real image path for testing
    test_image_paths = ['path_to_image1.jpg', 'path_to_image2.jpg']

    # Test generating captions
    generated_captions = generate_image_captions(test_image_paths)
    print("Generated captions:", generated_captions)

    # Here should be some meaningful test conditions
    # For now, we just check if we got a list of captions equal to the number of images
    assert len(generated_captions) == len(test_image_paths), 'Number of captions does not match number of images'

    # In a real scenario, you would check if the captions are valid and relevant to the images.
    print("Test passed!")

test_generate_image_captions()