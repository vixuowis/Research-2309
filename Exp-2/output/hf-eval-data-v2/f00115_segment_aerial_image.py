# function_import --------------------

from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image

# function_code --------------------

def segment_aerial_image(image_path):
    """
    This function uses a pre-trained model from Hugging Face Transformers to segment an aerial image.
    The model used is 'facebook/maskformer-swin-base-ade', which is trained on the ADE20k dataset and is suited for semantic segmentation tasks.

    Args:
        image_path (str): The path to the aerial image to be segmented.

    Returns:
        predicted_semantic_map: The segmented image, with different regions corresponding to the various objects and areas of interest.
    """
    image = Image.open(image_path)
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained('facebook/maskformer-swin-base-ade')
    inputs = feature_extractor(images=image, return_tensors='pt')
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-base-ade')
    outputs = model(**inputs)
    predicted_semantic_map = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

# test_function_code --------------------

def test_segment_aerial_image():
    """
    This function tests the 'segment_aerial_image' function by using a sample image.
    The test will pass if the function successfully segments the image without raising any exceptions.
    """
    sample_image_path = 'sample_aerial_image.jpg'
    # replace 'sample_aerial_image.jpg' with the path to your sample aerial image
    try:
        segmented_image = segment_aerial_image(sample_image_path)
        assert segmented_image is not None
        print('Test passed.')
    except Exception as e:
        print('Test failed. The following error occurred:')
        print(e)

# call_test_function_code --------------------

test_segment_aerial_image()