from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('facebook/blenderbot-90M')
model = AutoModelForCausalLM.from_pretrained('facebook/blenderbot-90M')

def respond_to_message(input_message):
    '''
    This function takes an input message as a string, tokenizes it, and passes it through a pre-trained conversational AI model.
    The model then generates a response, which is decoded and returned.
    Args:
    input_message (str): The input message that the model should respond to.
    Returns:
    str: The model's response.
    '''
    tokenized_input = tokenizer.encode(input_message + tokenizer.eos_token, return_tensors='pt')
    output = model.generate(tokenized_input, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, tokenized_input.shape[-1]:][0], skip_special_tokens=True)
    return response