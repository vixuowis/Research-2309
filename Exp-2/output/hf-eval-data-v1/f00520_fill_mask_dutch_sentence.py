from transformers import AutoTokenizer, AutoModel

# Function to fill in the missing word in a Dutch sentence
# using the 'GroNLP/bert-base-dutch-cased' model from Hugging Face Transformers

def fill_mask_dutch_sentence(input_sentence):
    """
    This function takes a Dutch sentence with a missing word (represented by [MASK])
    and uses the 'GroNLP/bert-base-dutch-cased' model to predict the missing word.
    
    Args:
    input_sentence (str): The Dutch sentence with a missing word (represented by [MASK])
    
    Returns:
    str: The Dutch sentence with the missing word filled in
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
    model = AutoModel.from_pretrained('GroNLP/bert-base-dutch-cased')
    
    # Tokenize the input sentence and get the model's output
    input_tokens = tokenizer(input_sentence, return_tensors='pt')
    outputs = model(**input_tokens)
    
    # Get the index of the masked token
    mask_token_index = torch.where(input_tokens['input_ids'][0] == tokenizer.mask_token_id)[0]
    
    # Get the model's prediction for the masked token
    mask_token_logits = outputs[0, mask_token_index, :]
    mask_token_id = torch.argmax(mask_token_logits).item()
    
    # Replace the masked token with the model's prediction
    input_tokens['input_ids'][0, mask_token_index] = mask_token_id
    
    # Decode the tokens to get the completed sentence
    completed_sentence = tokenizer.decode(input_tokens['input_ids'][0])
    
    return completed_sentence