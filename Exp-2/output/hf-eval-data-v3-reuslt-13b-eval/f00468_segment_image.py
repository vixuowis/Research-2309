# function_import --------------------

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_image(image_url):
    """
    Analyze an image of an urban scene to identify and separate regions with different semantics.

    Args:
        image_url (str): URL of the image to be analyzed.

    Returns:
        torch.Tensor: The output logits from the semantic segmentation model.

    Raises:
        Exception: If the image cannot be opened.
    """
    # Load the image and create a feature extractor
    try:
        im = Image.open(requests.get(image_url, stream=True).raw)
    except:
        raise Exception("Can't open image")
    feature_extractor = SegformerFeatureExtractor()

    # Create a model to analyze the semantic meaning of each pixel in the image
    model = SegformerForSemanticSegmentation.from_pretrained('nateraw/segformer-b0-finetuned-ade')
    model.eval()

    inputs = feature_extractor(images=im, return_tensors="pt")
    outputs = model(**inputs)

    # Convert the logits to predictions of each pixel and return them
    preds = torch.argmax(outputs.logits[0], dim=-1)
    return preds

# test_function_code --------------------

def test_segment_image():
    """
    Test the segment_image function.
    """
    test_image_url = 'https://placekitten.com/200/300'
    try:
        output = segment_image(test_image_url)
        assert output is not None, 'Output is None.'
        assert output.shape[0] == 1, 'Output shape is incorrect.'
    except Exception as e:
        print(f'Test failed with error: {e}')
    else:
        print('All tests passed.')


# call_test_function_code --------------------

test_segment_image()