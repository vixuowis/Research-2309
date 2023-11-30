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
    try:
      img = Image.open(image_path)
    except FileNotFoundError as e:
      raise Exception("Image file not found.") from e
    except AttributeError as e:
      raise TypeError('Invalid image path') from e 
      
    # Load the blip processor and model for CAPTION generation.
    processor = BlipProcessor.from_pretrained("models/blip-tokenizer")
    
    inputs = processor(
        text=[text],
        images=img,
        return_tensors="pt",
        padding='max_length',
        max_length=128,
        truncation=True,
    )

    # Load the blip model for CAPTION generation.
    model = BlipForConditionalGeneration.from_pretrained("models/blip")
    
    # Generate the caption using the model.
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        images=inputs.pixel_values,
        max_length=128,
        num_beams=4,
    )    

    return processor.batch_decode(outputs)[0] 

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