# requirements_file --------------------

!pip install -U transformers==4.16.2 Pillow==9.0.1 requests==2.27.1

# function_import --------------------

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_cityscape_image(image_url):
    '''
    Segment the image from the provided URL using a pre-trained SegFormer model.

    Parameters:
    - image_url (str): URL of the image to segment.

    Returns:
    tuple: (segmented image as a PIL Image, logits as a torch tensor).
    '''
    # Initialize the feature extractor and model
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')

    # Open the image
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Prepare the input tensors
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Perform inference
    outputs = model(**inputs)

    # Return the logits
    logits = outputs.logits
    return image, logits

# test_function_code --------------------

def test_segment_cityscape_image():
    print("Testing 'segment_cityscape_image' function.")
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image, logits = segment_cityscape_image(test_image_url)

    # Test case 1: Check if image is not None
    print("Testing case [1/3].")
    assert image is not None, "Test case [1/3] failed: The function returned None instead of an image."

    # Test case 2: Check if logits is not None
    print("Testing case [2/3].")
    assert logits is not None, "Test case [2/3] failed: The function returned None instead of logits."

    # Test case 3: Check if logits has the correct shape
    print("Testing case [3/3].")
    assert logits.dim() == 4, "Test case [3/3] failed: Logits tensor does not have 4 dimensions as expected."
    print("Testing completed successfully.")

# Run the test
test_segment_cityscape_image()