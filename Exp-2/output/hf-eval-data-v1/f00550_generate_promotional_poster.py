from transformers import pipeline

# Function to generate a promotional poster for a new line of summer clothing
# The function uses the Hugging Face's Realistic_Vision_V1.4 model to generate the image
# The function takes a prompt and a negative prompt as input
# The prompt describes the desired image
# The negative prompt describes what should not be in the image

def generate_promotional_poster(prompt: str, negative_prompt: str):
    # Create a pipeline for text-to-image using the Realistic_Vision_V1.4 model
    model = pipeline('text-to-image', model='SG161222/Realistic_Vision_V1.4')
    # Generate the image based on the prompt and constraints
    result = model(prompt, negative_prompt=negative_prompt)
    return result