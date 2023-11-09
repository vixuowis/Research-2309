from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def dialogue_response_generation(user_input):
    '''
    This function generates a dialogue response using the DialoGPT model from Hugging Face.
    The model is trained on 147M multi-turn dialogue from Reddit discussion thread.
    
    Args:
    user_input (str): The user's input to which the model will generate a response.
    
    Returns:
    str: The generated response from the model.
    '''
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')
    chat_history_ids = None
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)