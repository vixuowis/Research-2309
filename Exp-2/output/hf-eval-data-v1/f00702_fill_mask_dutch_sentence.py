from transformers import AutoTokenizer, AutoModel

def fill_mask_dutch_sentence(input_text):
    """
    This function fills in the missing word in a Dutch sentence using the GroNLP/bert-base-dutch-cased model from Transformers.
    
    Parameters:
    input_text (str): The input sentence with a missing word represented by '___'.
    
    Returns:
    str: The complete sentence with the missing word filled in.
    """
    tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
    model = AutoModel.from_pretrained('GroNLP/bert-base-dutch-cased')
    input_tokens = tokenizer.encode(input_text, return_tensors="pt")
    mask_position = input_tokens.tolist()[0].index(tokenizer.mask_token_id)
    output = model(input_tokens)
    prediction = output.logits.argmax(dim=2)[0].item()
    predicted_word = tokenizer.convert_ids_to_tokens(prediction)
    filled_sentence = input_text.replace("___", predicted_word)
    return filled_sentence