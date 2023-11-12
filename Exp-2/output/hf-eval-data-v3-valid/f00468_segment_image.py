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
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b2-finetuned-cityscapes-1024-1024')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b2-finetuned-cityscapes-1024-1024')

    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise Exception('Unable to open image.') from e

    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    return outputs.logits

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