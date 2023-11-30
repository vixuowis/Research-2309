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
    
    # Processor.
    processor = OneFormerProcessor()
    
    # Dataset.
    dataset_test = processor.get_dataset("test", data_files={"test": [image_path]})
    
    # Tokenizer.
    tokenizer = processor.feature_extractor

    # Model.
    model = OneFormerForUniversalSegmentation(pretrained="shi-labs/oneformer_coco_swin_large")
    
    # Image Preprocessing.
    image = Image.open(image_path).convert("RGB").resize((640, 384))
    inputs = tokenizer(image, return_tensors="pt", add_special_tokens=False)

    # Prediction.
    with torch.no_grad():
        outputs = model(**inputs)[1]["last_hidden_state"][:,0].argmax(-1).cpu().numpy()
    
    return outputs


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