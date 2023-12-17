# requirements_file --------------------

!pip install -U transformers Pillow requests

# function_import --------------------

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_urban_scene_image(image_url):
    """
    Segments an urban scene image using a pre-trained SegFormer model.
    
    :param image_url: URL of the image to be segmented.
    
    :return: An object containing the logits representing the segmented image.
    """
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b2-finetuned-cityscapes-1024-1024')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b2-finetuned-cityscapes-1024-1024')
    
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    
    return outputs.logits

# test_function_code --------------------

def test_segment_urban_scene_image():
    print("Testing segment_urban_scene_image started.")
    example_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    
    # Test case 1: Check if the function is callable with an image URL
    print("Testing case [1/2] started.")
    try:
        segment_urban_scene_image(example_image_url)
        print("Test case [1/2] passed.")
    except Exception as e:
        assert False, f"Test case [1/2] failed: {e}"
    
    # Test case 2: Check if the function returns logits
    print("Testing case [2/2] started.")
    logits = segment_urban_scene_image(example_image_url)
    assert logits is not None, "Test case [2/2] failed: The function did not return any output."
    assert logits.dim() == 4, f"Test case [2/2] failed: The output logits should have 4 dimensions, got {logits.dim()} instead."
    print("Testing segment_urban_scene_image finished.")

# Run the test function
test_segment_urban_scene_image()