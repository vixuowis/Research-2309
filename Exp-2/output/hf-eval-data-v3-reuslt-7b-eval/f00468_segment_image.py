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

    # Initialize a feature extractor for the SegFormer model.

    feature_extractor = SegformerFeatureExtractor()
    
    # Create a Segmentation pipeline using the Semantic Segmentation head of the SegFormer model.
    
    segmenter = SegformerForSemanticSegmentation(feature_extractor, 1209)
    
    try:
        # Open the image from a URL.
        
        pil_image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        print("Could not open", image_url,"Exception:",e)
    
    # Resize the image to 512 x 512 pixels and convert it to a tensor.
        
    image = feature_extractor(pil_image, return_tensors="pt")["pixel_values"].unsqueeze(0)
    
    # Run inference on the image with SegFormer model to produce segmentation logits.
    
    outputs = segmenter(images=image)
    
    # Get a softmax probability of the pixel-wise labels from the output logits.
        
    probabilities = outputs.logits.softmax(-1)[0] 

    return probabilities

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