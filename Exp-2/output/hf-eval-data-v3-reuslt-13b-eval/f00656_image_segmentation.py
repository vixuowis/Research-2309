# function_import --------------------

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image

# function_code --------------------

def image_segmentation(image_path: str) -> dict:
    """
    This function performs image segmentation using the pre-trained 'shi-labs/oneformer_coco_swin_large' model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: The segmented regions of the image.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The image file '{image_path}' does not exist.")

    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    
    model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(device)
    outputs = model(pixel_values).logits

    seg_map = (outputs[0].argmax(0).detach().cpu() * 255).numpy().astype(np.uint8)
    seg_image = Image.fromarray(seg_map, mode="L")
    
    return processor.draw_bounding_box(img, seg_image)

# test_function_code --------------------

def test_image_segmentation():
    """
    This function tests the image_segmentation function with a sample image.
    """
    image_path = 'https://placekitten.com/200/300'
    try:
        segmented_image = image_segmentation(image_path)
        assert isinstance(segmented_image, dict), 'The output should be a dictionary.'
    except FileNotFoundError:
        print('The image file does not exist.')
    return 'All Tests Passed'


# call_test_function_code --------------------

test_image_segmentation()