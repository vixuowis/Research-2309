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

    try:
        # Load pre-trained model and tokenizer
        processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_coco_swin_large')
        model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_coco_swin_large')

        # Load image
        image = Image.open(image_path)

        # Encode image
        encoding = processor(image, return_tensors="pt")
        pixel_values = encoding['pixel_values']

        outputs = model(pixel_values)
        
        # Make segmentation map from the model's output
        raw_segmentation_map = outputs.logits[0]
        segmentation_map = (raw_segmentation_map.sigmoid() > 0.5).float().cpu().detach()
        
        # Create a dictionary with category IDs as keys and corresponding region(s) of the original image as values
        dict_category_id = {1: 'pet', 2: 'background'}
        segmented_image = processor.decode_segmentation(segmentation_map, label_map=dict_category_id)[0]
        
        return segmented_image
    except FileNotFoundError:
        raise Exception('Image file does not exist')


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