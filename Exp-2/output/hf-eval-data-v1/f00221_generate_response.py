from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')

def generate_response(user_input, chat_history):
    '''
    This function generates a response from the DialoGPT model given a user input and chat history.
    Args:
    user_input (str): The input from the user.
    chat_history (torch.Tensor): The chat history.
    Returns:
    str, torch.Tensor: The generated response and the updated chat history.
    '''
    # Encode the user input and add the end of sentence token
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    # Concatenate the chat history with the user input
    chat_history = torch.cat([chat_history, input_ids], dim=-1) if chat_history is not None else input_ids
    # Generate a response from the model
    outputs = model.generate(chat_history, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    # Decode the response
    response = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, outputs