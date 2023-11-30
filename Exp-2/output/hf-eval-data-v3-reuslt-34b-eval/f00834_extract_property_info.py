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

    if os.path.isfile(image_path) == False:
        raise FileNotFoundError("The image file was not found")
    
    model = LayoutLMv3ForQuestionAnswering.from_pretrained('microsoft/layoutlmv3-base')
    labels = "labels.txt"
    
    if os.path.isfile(labels) == False:
        raise FileNotFoundError("Labels file was not found")
    
    with open(labels, 'r', encoding='utf-8') as f:
        context = [line.strip() for line in f]
        
    questions = ["What is the address?", "What is the phone number?"]
    
    if os.path.isfile("config.json") == False and os.path.isfile("config.py") == False:
        raise FileNotFoundError('The model config file was not found')
        
    elif os.path.isfile("config.json"):
        config_dict = json.load(open("config.json"))
    
    else:
        with open("config.py", 'r', encoding='utf-8') as f:
            config_code = f.read()
            
        config_dict = literal_eval(config_code)
        
    model.load_blobs_from_config_dict(config_dict)
    
    image = Image.open(image_path).convert("RGB")
    pixel_values = resize_transform(np.array(image)).unsqueeze(0)
    
    answers = []
    
    for question in questions:
        inputs = model.generate_inputs(pixel_values, question, context, return_tensors="pt")
        
        input_ids = inputs["input_ids"].to(model.device)
        pixel_values = inputs["pixel_values"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
    
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask).logits
        outputs_idx = outputs.

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