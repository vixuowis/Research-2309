# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_image(img_url: str):
    """
    Classify an image using a pretrained CLIP model.

    Args:
        img_url (str): The URL of the image to classify.

    Returns:
        dict: A dictionary where keys are labels and values are probabilities.
    """
    # Download the image from the given URL.
    r = requests.get(img_url)
    pil_image = Image.open(BytesIO(r.content))
    
    # Load CLIP model and processor.
    model_name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Process the image and get labels and scores as a dictionary.
    inputs = processor(pil_image, return_tensors='pt')
    outputs = model(**inputs).logits_per_image[0].softmax(-1)
    score_dict = {label: score for label, score in zip(processor.tokenizer.get_labels(), outputs)}
    
    # Remove the null class (scores of which are close to 0) and return the dictionary.
    return {k: v for k, v in sorted(score_dict.items(), key=lambda item: -item[1]) if k != "__null__"}


# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function.
    """
    img_url = 'https://placekitten.com/200/300'
    result = classify_image(img_url)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(['residential area', 'playground', 'stadium', 'forest', 'airport'])
    assert all(0 <= v <= 1 for v in result.values())
    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_image()