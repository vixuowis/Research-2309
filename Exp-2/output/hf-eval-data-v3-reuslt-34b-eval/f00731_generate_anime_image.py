# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch
import os

# function_code --------------------

def generate_anime_image(prompt: str, negative_prompt: str, save_path: str = './result.jpg'):
    '''
    Generate an anime image based on the given prompt and negative_prompt.

    Args:
        prompt (str): The description of the desired character appearance.
        negative_prompt (str): The features that should be excluded from the generated image.
        save_path (str, optional): The path to save the generated image. Defaults to './result.jpg'.

    Returns:
        None
    '''
    pipeline = StableDiffusionPipeline(steps=60)
    # TODO: make these paths configurable as args or environment variables
    pipeline.load_model('./models/pytorch_diffusion_uncond_finetune_008100.pt')
    pipeline.alias_maker = AliasMaker(path='./models/256x256_diffusion_unconditional_finetune_008100_epoch60_2021_12_31_17_56_41.pt')
    prompt = pipeline.alias(prompt)
    negative_prompt = pipeline.alias(negative_prompt)
    print('---------------------', prompt)
    
    img = pipeline.run(prompt=prompt, negative_prompt=negative_prompt, batch_size=1, lr=.02)
    
    img = (img + 1)*127.5
    img = torch.permute(torch.clip(img[:3], min=-1, max=1), (1, 2, 0)).cpu().detach().numpy()
    os.makedirs('./result', exist_ok=True)
    cv2.imwrite(save_path, img)

# test_function_code --------------------

def test_generate_anime_image():
    '''
    Test the function generate_anime_image.
    '''
    prompt = 'anime, masterpiece, high quality, 1girl, solo, long hair, looking at viewer, blush, smile, bangs, blue eyes, skirt, medium breasts, iridescent, gradient, colorful'
    negative_prompt = 'simple background, duplicate, retro style, low quality, lowest quality, 1980s, 1990s, 2000s, 2005 2006 2007 2008 2009 2010 2011 2012 2013, bad anatomy, bad proportions, extra digits, lowres, username, artist name, error, duplicate, watermark, signature, text, extra digit, fewer digits, worst quality, jpeg artifacts, blurry'
    generate_anime_image(prompt, negative_prompt)
    assert os.path.exists('./result.jpg'), 'Image not generated'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_anime_image()