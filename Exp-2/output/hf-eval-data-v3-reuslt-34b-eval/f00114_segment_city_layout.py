# function_import --------------------

from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from PIL import Image
import requests

# function_code --------------------

def segment_city_layout(image_url):
    """
    This function takes an image URL of a city layout and returns the segmented image using a pre-trained model.
    The model used is 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024' from Hugging Face Transformers.

    Args:
        image_url (str): The URL of the city layout image.

    Returns:
        torch.Tensor: The segmented image.

    Raises:
        requests.exceptions.RequestException: If there is a problem with the network connection.
        requests.exceptions.HTTPError: If there is an HTTP error.
        requests.exceptions.Timeout: If the request times out.
        requests.exceptions.TooManyRedirects: If the request exceeds the configured number of maximum redirections.
    """
    
    # load model and feature extractor
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    feature_extractor = SegformerFeatureExtractor.from_pretrained(
        'nvidia/segformer-b5-finetuned-cityscapes-1024-1024', image_size=512)  # set image_size for segmentation
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load and preprocess the image
    try:
        response = requests.get(image_url)
        
        try:
            img = Image.open(BytesIO(response.content))  # load image
            
            encoded_inputs = feature_extractor(images=img, return_tensors='pt')  # preprocess for segmentation
    
            pixel_values = encoded_inputs['pixel_values'].to(device)  # move to GPU/CPU device
            
        except OSError:
            raise Exception("Problem with image file.")
        
    except requests.exceptions.RequestException as e:
        raise e
    
    # run the model on the input pixel_values tensor
    outputs = model(pixel_value=pixel_values)
    
    # save segmented image
    segemented_image = outputs['logits'].cpu().detach()  # move segmented image to CPU
    
    return segemented_image

# test_function_code --------------------

def test_segment_city_layout():
    """Tests the `segment_city_layout` function."""
    # Test with a city layout image
    image_url = 'https://placekitten.com/200/300'
    output = segment_city_layout(image_url)
    assert output is not None, 'The output should not be None.'
    assert output.shape[0] == 1, 'The output shape should be (1, num_classes, height, width).'

    # Test with another city layout image
    image_url = 'https://placekitten.com/200/300'
    output = segment_city_layout(image_url)
    assert output is not None, 'The output should not be None.'
    assert output.shape[0] == 1, 'The output shape should be (1, num_classes, height, width).'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_segment_city_layout()