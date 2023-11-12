# function_import --------------------

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# function_code --------------------

model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
feature_extractor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
tokenizer = AutoTokenizer.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
max_length = 16
num_beams = 4
gen_kwargs = {'max_length': max_length, 'num_beams': num_beams}

def generate_captions(image_paths):
    '''
    Args:
        image_paths (list): List of image paths
    Returns:
        list: List of generated captions for each image
    '''
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != 'RGB':
            i_image = i_image.convert('RGB')
        images.append(i_image)
    pixel_values = feature_extractor(images=images, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

# test_function_code --------------------

def test_generate_captions():
    '''
    Test function for generate_captions
    '''
    image_paths = ['https://placekitten.com/200/300', 'https://placekitten.com/200/300', 'https://placekitten.com/200/300']
    captions = generate_captions(image_paths)
    assert isinstance(captions, list), 'Output should be a list'
    assert len(captions) == len(image_paths), 'Number of captions should be equal to number of images'
    for caption in captions:
        assert isinstance(caption, str), 'Each caption should be a string'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_captions()