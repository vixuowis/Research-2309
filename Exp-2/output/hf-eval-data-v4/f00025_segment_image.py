# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_image(url):
    # Load the image from the provided URL
    image = Image.open(requests.get(url, stream=True).raw)

    # Initialize the image processor
    processor = MaskFormerImageProcessor.from_pretrained('facebook/maskformer-swin-large-ade')

    # Preprocess the image for the model
    inputs = processor(images=image, return_tensors='pt')

    # Load the pretrained instance segmentation model
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-large-ade')

    # Perform prediction
    outputs = model(**inputs)

    # Process the outputs to get the segmentation maps
    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    return predicted_semantic_map

# test_function_code --------------------

def test_segment_image():
    print("Testing started.")
    # Test with a sample image URL
    sample_url = 'https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg'

    # Execute the segmentation function
    predicted_semantic_map = segment_image(sample_url)

    # Check if the segmentation map is not empty
    assert predicted_semantic_map is not None, "Test failed: The returned segmentation map is None."
    print("Testing finished.")

# Run the test
test_segment_image()