# function_import --------------------

from diffusers import DDPMPipeline
from PIL import Image

# function_code --------------------

def generate_minecraft_skin():
    """
    This function generates a Minecraft skin image using a pre-trained model from Hugging Face Transformers.

    Returns:
        PIL.Image.Image: The generated Minecraft skin image in RGBA format.
    """    
    model_name = "microsoft/DPRQuestionAnswering-t4"
    diffuser = DDPMPipeline(model_name)

    question_text = "What is the name of this character?"
    context_text = (
        "After spending years studying robotics, he returns to his childhood home in Susanoo to help his ailing father. " 
        + "But what he discovers there may change his life forever."
    )
    image = diffuser(question=question_text, context=context_text)["image"]
    return Image.open(io.BytesIO(image))

# test_function_code --------------------

def test_generate_minecraft_skin():
    """
    This function tests the generate_minecraft_skin function by checking the type and mode of the returned image.
    """
    image = generate_minecraft_skin()
    assert isinstance(image, Image.Image), 'The returned object is not a PIL image.'
    assert image.mode == 'RGBA', 'The image is not in RGBA format.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_minecraft_skin()