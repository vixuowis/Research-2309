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

def predict_caption(image_path):
    '''
    Args:
        image_path: A string that represents the path of the image file.
    Returns:
        A string that represents the predicted caption of the image.
    '''
    input_image = Image.open(image_path)
    if input_image.mode != 'RGB':
        input_image = input_image.convert(mode='RGB')

    pixel_values = feature_extractor(images=[input_image], return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption.strip()

# test_function_code --------------------

def test_predict_caption():
    assert predict_caption('https://placekitten.com/200/300') is not None
    assert isinstance(predict_caption('https://placekitten.com/200/300'), str)
    assert len(predict_caption('https://placekitten.com/200/300')) > 0
    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_caption()