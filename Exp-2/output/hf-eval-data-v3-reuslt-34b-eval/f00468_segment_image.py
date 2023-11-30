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
    
    # Download the image and save it as `image_pil`.
    try:
        response = requests.get(image_url)
        image_pil = Image.open(BytesIO(response.content))

    except Exception as error:
        raise Exception(error)
    
    # Load the feature extractor and the model.
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    
    # Encode the input image and generate an output prediction.
    inputs = feature_extractor(images=image_pil, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits[0]

    return logits

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