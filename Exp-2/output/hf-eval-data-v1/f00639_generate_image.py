from diffusers import StableDiffusionPipeline
import torch

# Function to generate an image based on a text prompt using a pre-trained model
# @param prompt: The text prompt to generate the image from
# @return: The generated image

def generate_image(prompt):
    # Load the pre-trained model
    model_id = 'dreamlike-art/dreamlike-photoreal-2.0'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    
    # Generate the image from the text prompt
    generated_image = pipe(prompt).images[0]
    
    # Save the image
    generated_image.save('result.png')
    
    return generated_image