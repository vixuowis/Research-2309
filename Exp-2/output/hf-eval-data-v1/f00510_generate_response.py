from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelWithLMHead.from_pretrained('output-small')

# Function to generate response
def generate_response(user_input):
    '''
    This function takes user input as argument and generates a response using a pre-trained model.
    The model is trained on the speech of a game character, Joshua from The World Ends With You.
    '''
    # Encode the user input
    user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    # Concatenate the previous chatbot response with the user input
    bot_input_ids = torch.cat([chat_history_ids, user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    # Generate a response using the model
    chat_history_ids = model.generate(bot_input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3, do_sample=True, top_k=100, top_p=0.7, temperature = 0.8)
    # Decode the model output to provide the AI response
    ai_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return ai_response