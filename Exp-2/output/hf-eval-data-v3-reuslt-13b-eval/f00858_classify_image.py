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

    # Load a pretrained CLIP model.
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Get the image from a URL.
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))

    # Convert it into a tensor to be passed through the model.
    inputs = processor(
        text=["a picture"], images=img, return_tensors="pt", padding=True
    )

    # Load another pretrained CLIP model with labels for ImageNet.
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # shape = [batch_size]
    probs = logits_per_image.softmax(dim=-1).tolist()[0]  # shape = [num_labels]

    # Convert the probabilities into a dictionary where keys are labels and values are probabilities.
    model_url = "https://huggingface.co/openai/clip-vit-base-patch32"
    resp = requests.get(f"{model_url}/api/info")
    
    labels = []
    for label in resp.json()["metadata"]["result_labels"]:
        labels.append(label[10:])  # Strip "attribute_" from the beginning of each label.

    return dict(zip(labels, probs))


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