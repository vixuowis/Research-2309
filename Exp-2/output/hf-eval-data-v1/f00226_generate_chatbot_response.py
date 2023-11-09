from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch

# Function to generate chatbot response
def generate_chatbot_response(prompt):
    '''
    This function generates a chatbot response for a given prompt using the pretrained model 'facebook/opt-66b'.
    Args:
    prompt (str): The prompt for the chatbot.
    Returns:
    list: A list of generated responses.
    '''
    # Load the model
    model = AutoModelForCausalLM.from_pretrained('facebook/opt-66b', torch_dtype=torch.float16).cuda()
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-66b', use_fast=False)
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    # Set a random seed for reproducibility
    set_seed(32)
    # Generate a response
    generated_ids = model.generate(input_ids, do_sample=True, num_return_sequences=5, max_length=10)
    # Decode the generated responses
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses