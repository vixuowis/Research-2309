import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Define the model name
MODEL_NAME = 'cointegrated/rut5-base-absum'

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Move the model to GPU if available
model.cuda()

# Set the model to evaluation mode
model.eval()

# Define the function to summarize Russian text
def summarize_russian_text(text, max_length=1000, num_beams=3, do_sample=False, repetition_penalty=10.0, **kwargs):
    """
    This function takes a Russian text as input and returns a brief summary of the text.
    It uses the T5ForConditionalGeneration model from Hugging Face Transformers.
    
    Parameters:
    text (str): The Russian text to be summarized.
    max_length (int): The maximum length of the summary. Default is 1000.
    num_beams (int): The number of beams for beam search. Default is 3.
    do_sample (bool): Whether to do sampling. Default is False.
    repetition_penalty (float): The penalty for repetition. Default is 10.0.
    kwargs: Other optional arguments.
    
    Returns:
    str: The summary of the input text.
    """
    # Tokenize the input text and create input tensors
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    
    # Generate a summary using the model's 'generate' function
    with torch.inference_mode():
        out = model.generate(x, max_length=max_length, num_beams=num_beams, do_sample=do_sample, repetition_penalty=repetition_penalty, **kwargs)
    
    # Decode the generated text to produce the final summary
    return tokenizer.decode(out[0], skip_special_tokens=True)