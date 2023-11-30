# function_import --------------------

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_image(image_url):
    '''
    Classify the image using Vision Transformer (ViT).

    Args:
        image_url (str): The url of the image to be classified.

    Returns:
        str: The predicted class of the image.

    Raises:
        OSError: If there is a problem with the network connection or the image file.
    '''

    # load model and tokenizer
    model_id = 'google/vit-base-patch16-224'
    processor = ViTImageProcessor(img_size=224)
    model = ViTForImageClassification.from_pretrained(model_id, num_labels=1000)
    
    # prepare image and input
    url = requests.get(image_url)
    img = Image.open(BytesIO(url.content))
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape = [1, 1000]
    
    # predict class and confidence of top prediction
    probs = torch.softmax(logits[0], dim=-1).tolist()  # shape = [1000,]
    predicted_index = logits.argmax().item()
    predicted_class = model.config.id2label[predicted_index]
    
    return (predicted_class, probs[predicted_index])

# test_function_code --------------------

def test_classify_image():
    '''
    Test the classify_image function.
    '''
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    predicted_class = classify_image(test_image_url)
    assert isinstance(predicted_class, str), 'The predicted class should be a string.'
    print('All Tests Passed')


# call_test_function_code --------------------

if __name__ == '__main__':
    test_classify_image()