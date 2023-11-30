# function_import --------------------

import torch
import requests
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

# function_code --------------------

def extract_captions(image_url):
    """
    Extract captions from an image using a pre-trained model from Hugging Face.

    Args:
        image_url (str): URL of the image to extract captions from.

    Returns:
        list: A list of generated captions.

    Raises:
        OSError: If there is an error in loading the image or the pre-trained model.
    """

    try:
        # load images and tokenizer
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        tokenizer = AutoTokenizer.from_pretrained('allenai/led-large-16384-arxiv')

        # load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VisionEncoderDecoderModel.from_pretrained('allenai/led-large-16384-arxiv').to(device)
    except OSError as e:
        print("OSError in load pre-trained model or tokenizer.")
        print(e)
        raise e

    # load image from url, resize the image and convert to tensor
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        resized_img = feature_extractor(images=img, return_tensors="pt")['pixel_values']
    except OSError as e:
        print("OSError in load or resize the image.")
        print(e)
        raise e
    
    # generate captions for the images using the pre-trained model
    try:
        input = resized_img.to(device=device, dtype=torch.float32)
        output_ids = model.generate(input, max_length=50, do_sample=True, top_p=.95, num_return_sequences=1)[0]
    except RuntimeError as e:
        print("Runtime Error in generating captions.")
        print(e)
        raise e
    
    # decode the generated tokens into human-readable caption
    return tokenizer.decode(output_ids, skip_special_tokens=True)

# test_function_code --------------------

def test_extract_captions():
    """
    Test the extract_captions function with a few test cases.
    """
    test_cases = [
        'http://images.cocodataset.org/val2017/000000039769.jpg',
        'https://placekitten.com/200/300',
        'https://placekitten.com/400/600'
    ]
    for url in test_cases:
        captions = extract_captions(url)
        assert isinstance(captions, list), 'The output should be a list.'
        assert all(isinstance(caption, str) for caption in captions), 'All elements in the output list should be strings.'
    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_extract_captions())