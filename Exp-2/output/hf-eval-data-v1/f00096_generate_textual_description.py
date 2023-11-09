from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Function to generate textual descriptions for images and videos
# Uses the pre-trained GIT model from Hugging Face Transformers
# The model is specifically designed for this task and has been fine-tuned on the TextCaps dataset
# The function takes as input an encoded image and a text prompt, and returns a generated text description

def generate_textual_description(encoded_image, text):
    # Load the pre-trained GIT model
    model = AutoModelForCausalLM.from_pretrained('microsoft/git-large-textcaps')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/git-large-textcaps')

    # Prepare the image and text inputs
    # Encode the text tokens and concatenate them with the image tokens
    input_ids = tokenizer(text, return_tensors="pt", padding=True).input_ids
    prompt_length = len(input_ids[0])

    # Concatenate the encoded image and text tokens
    input_ids = torch.cat([encoded_image, input_ids], dim=1)

    # Run the model to generate a text description
    output = model.generate(input_ids, max_length=prompt_length + 20)

    # Decode the output to get the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text