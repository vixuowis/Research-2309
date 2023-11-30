# function_import --------------------

import os
from transformers import LayoutLMv3ForQuestionAnswering

# function_code --------------------

def extract_property_info(image_path):
    """
    Extracts property information from a scanned image using LayoutLMv3ForQuestionAnswering model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The extracted property information.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    
    # Check if the input image is a valid file
    if not os.path.isfile(image_path):
        raise FileNotFoundError("Input image path is invalid or does not point to an existing file.")
        
    # Set the model name and load it from Hugging Face
    model_name = "microsoft/layoutlmv3-base"  # The name of the model to use.
    layout_model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_name)
    
    # Read the image and extract property information from it
    with open(image_path, "rb") as file:
        image_bytes = file.read()
        
        inputs = layout_model.processor(image_bytes, padding="max_length", truncation=True)
        input_ids = inputs["input_ids"][0]
        outputs = layout_model(**{k: v[:1] for k, v in inputs.items()})
        
        answer = layout_model.processor.tokenizer.decode(outputs.start_logits[0].argmax())
    
    return answer  # Return the extracted property information as a string

# test_function_code --------------------

def test_extract_property_info():
    """
    Tests the extract_property_info function.
    """
    # Test with a non-existing image file
    try:
        extract_property_info('non_existing_file.jpg')
    except FileNotFoundError as e:
        assert str(e) == 'non_existing_file.jpg does not exist'

    # TODO: Add more test cases

    return 'All Tests Passed'


# call_test_function_code --------------------

test_extract_property_info()