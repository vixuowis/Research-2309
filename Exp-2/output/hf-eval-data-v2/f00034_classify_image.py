# function_import --------------------

from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# function_code --------------------

def classify_image(image_url, texts):
    """
    Classify an image based on semantic similarity to the given texts using a pre-trained Chinese CLIP model.

    Args:
        image_url (str): The URL of the image to be classified.
        texts (list of str): The texts to be compared with the image for classification.

    Returns:
        probs (torch.Tensor): The probabilities of each text being the correct classification for the image.
    """
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=texts, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return probs

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function with a sample image and texts.
    """
    image_url = 'https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg'
    texts = ['皮卡丘', '超梦', '杰尼龟']
    probs = classify_image(image_url, texts)
    assert probs is not None, 'The function should return a value.'
    assert probs.size(0) == len(texts), 'The size of the output tensor should match the number of texts.'

# call_test_function_code --------------------

test_classify_image()