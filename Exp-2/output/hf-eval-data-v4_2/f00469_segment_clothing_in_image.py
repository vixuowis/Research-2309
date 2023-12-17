# requirements_file --------------------

!pip install -U transformers PIL requests matplotlib torch

# function_import --------------------

from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import torch.nn.functional as F
import matplotlib.pyplot as plt

# function_code --------------------

def segment_clothing_in_image(image_url):
    """Segments clothing items in an image using a pretrained Segformer model.

    Args:
        image_url (str): URL or local path of the image to be segmented.

    Returns:
        plt.Figure: A matplotlib figure illustrating the segmented clothing items.

    Raises:
        ValueError: If image_url is not reachable or invalid.
    """
    extractor = AutoFeatureExtractor.from_pretrained('mattmdjaga/segformer_b2_clothes')
    model = SegformerForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes')
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError(f'Unable to load image from the provided URL: {str(e)}')

    inputs = extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = F.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0]

    fig = plt.figure()
    plt.imshow(pred_seg)
    plt.axis('off')
    return fig

# test_function_code --------------------

def test_segment_clothing_in_image():
    print("Testing started.")

    # Testing with an example image URL
    print("Testing case [1/1] started.")
    try:
        fig = segment_clothing_in_image('https://example.com/image.jpg')
        assert isinstance(fig, plt.Figure), "Returned result is not a matplotlib figure."
    except Exception as e:
        assert False, f"Test case [1/1] failed: {str(e)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_segment_clothing_in_image()