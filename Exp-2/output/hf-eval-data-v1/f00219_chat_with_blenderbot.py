from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
import torch


def chat_with_blenderbot(message):
    """
    This function uses the BlenderbotForConditionalGeneration model from the transformers library to generate a reply to a given message.
    The model is pretrained on the 'facebook/blenderbot-400M-distill' dataset.
    
    Args:
        message (str): The message to which the bot should respond.
    
    Returns:
        str: The bot's response.
    """
    # Load the model and tokenizer
    model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')
    tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
    
    # Tokenize the message and convert it to tensors
    inputs = tokenizer(message, return_tensors='pt')
    
    # Generate a response
    outputs = model.generate(**inputs)
    
    # Decode the response and return it
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response