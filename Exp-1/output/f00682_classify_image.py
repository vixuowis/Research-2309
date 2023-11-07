from typing import *
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def classify_image(categories, image_url):
    """
    Classify the given image into a single category from the provided list.
    
    Args:
        categories (list): A list of categories.
        image_url (str): The URL of the image to be classified.
    
    Returns:
        str: The predicted category of the image.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prompt = [f"Instruction: Classify the following image into a single category from the following list: {categories}.",
              image_url,
              "Category:"]
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    bad_words_ids = tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = model.generate(**inputs, max_new_tokens=6, bad_words_ids=bad_words_ids)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_text[0]

