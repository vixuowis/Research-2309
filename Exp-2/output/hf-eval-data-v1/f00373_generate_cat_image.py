from diffusers import DDPMPipeline


def generate_cat_image():
    '''
    This function uses the DDPMPipeline from the diffusers package to generate cat-themed images.
    The function uses a pre-trained model 'google/ddpm-ema-cat-256' which is specifically trained for generating cat-related images.
    The generated image is then saved for further use.
    '''
    # Load the pre-trained model
    ddpm = DDPMPipeline.from_pretrained('google/ddpm-ema-cat-256')
    
    # Generate the image
    image = ddpm().images[0]
    
    # Save the generated image
    image.save('cat_character_image.png')
    
    return image