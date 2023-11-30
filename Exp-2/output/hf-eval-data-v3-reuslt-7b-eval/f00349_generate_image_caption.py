# function_import --------------------

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# function_code --------------------

def generate_image_caption(image_path: str, text: str = 'product photography') -> str:
    """
    Generate descriptive captions for photographs related to the products using Hugging Face Transformers.

    Args:
        image_path (str): The path to the image file.
        text (str, optional): A short text that provides some context to the photograph. Defaults to 'product photography'.

    Returns:
        str: The generated caption for the input image.
    """
    # Create processor and model
    processor = BlipProcessor.from_pretrained('facebook/blip-large-coco-airbnb')
    model = BlipForConditionalGeneration.from_pretrained('facebook/blip-large-coco-airbnb').to("cuda") # Change to "cpu" if you do not have a GPU
    
    # Prepare image for processing
    pil_img = Image.open(image_path).convert("RGB").resize((384, 384), resample=0)
    img = processor(images=pil_img, return_tensors="pt")['pixel_values'].to("cuda")
    
    # Prepare text for processing
    input_ids = processor.tokenizer([text], truncation='longest_first', max_length=128, padding='max_length').to("cuda")["input_ids"]
        
    # Generate caption with model
    output = model.generate(
        input_ids, 
        attention_mask=(input_ids > 0),
        max_length=512, 
        num_beams=4, 
        early_stopping=True)
    
    return processor.tokenizer.decode(output[0], skip_special_tokens=True)

# test_function_code --------------------

def test_generate_image_caption():
    """
    Test the function generate_image_caption.
    """
    assert isinstance(generate_image_caption('test_image.jpg', 'product photography'), str)
    assert isinstance(generate_image_caption('test_image.jpg', 'landscape photography'), str)
    assert isinstance(generate_image_caption('test_image.jpg', 'portrait photography'), str)
    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_generate_image_caption())