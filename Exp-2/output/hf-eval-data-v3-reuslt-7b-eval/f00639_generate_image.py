# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch
import os

# function_code --------------------

def generate_image(prompt: str, model_id: str = 'dreamlike-art/dreamlike-photoreal-2.0', device: str = 'cuda') -> None:
    """
    Generate an image based on the given text prompt using a pre-trained model.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_id (str, optional): The ID of the pre-trained model to use. Defaults to 'dreamlike-art/dreamlike-photoreal-2.0'.
        device (str, optional): The device to run the model on. Defaults to 'cuda'.

    Returns:
        None. The function saves the generated image as 'result.png' in the current directory.
    """    
    # load model
    if os.path.exists(f'{model_id}.ckpt'):
        
        # load checkpoint and config from file
        pipeline = StableDiffusionPipeline.load_from_checkpoint(os.getcwd() + f'/{model_id}.ckpt', model_type="openai")    
    else:
        # download model and config
        pipeline = StableDiffusionPipeline("openai", model_id=model_id)
        
        # save checkpoint and config to file
        pipeline.save_pretrained(os.getcwd())
    
    # generate image
    img, text_enc, clip_img = pipeline._prepare_inputs_for_generation(text=prompt, images=[], return_tensors='pt', 
                                                        template=None, max_target_length=512)
    
    # move to device and zero gradients
    img = img.to(device)
    text_enc = text_enc.to(device)
    clip_img = clip_img.to(device)

    with torch.no_grad():
        generated_images, losses = pipeline._model.generate_images(text_enc=text_enc, clip_img=clip_img, 
                                image_size=(256, 256), num_cutouts=10, cutn=32)
        
        generated_image = generated_images[0] / 2 + 0.5
    
    torchvision.utils.save_image(generated_image, 'result.png')

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.
    """
    generate_image('astronaut playing guitar in space')
    assert os.path.exists('result.png'), 'Image not generated'
    os.remove('result.png')
    generate_image('a cat sitting on a tree')
    assert os.path.exists('result.png'), 'Image not generated'
    os.remove('result.png')
    generate_image('a beautiful sunset over the ocean')
    assert os.path.exists('result.png'), 'Image not generated'
    os.remove('result.png')
    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_generate_image())