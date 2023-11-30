# function_import --------------------

from diffusers import DDPMPipeline
import os

# function_code --------------------

def generate_image(model_id: str, save_path: str) -> None:
    '''
    Generate an image using a pretrained model and save it to a specified path.

    Args:
        model_id (str): The ID of the pretrained model to use for image generation.
        save_path (str): The path where the generated image will be saved.

    Returns:
        None
    '''

    # load pipeline
    ddpmp = DDPMPipeline.load(model_id)

    # generate an image
    imgs, losses = ddpmp.generate()

    # save generated image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for i, img in enumerate(imgs):
        if i == 1:
            break
        imsave(f'{save_path}_{i}.jpg', (255*np.clip(img[0], 0., 1.).astype('uint8')).astype('float32'))


# test_function_code --------------------

def test_generate_image():
    '''
    Test the generate_image function.
    '''
    model_id = 'google/ddpm-church-256'
    save_path = 'test_image.png'
    generate_image(model_id, save_path)
    assert os.path.exists(save_path), 'Image not saved correctly'
    os.remove(save_path)
    assert not os.path.exists(save_path), 'Image not removed correctly'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_image()