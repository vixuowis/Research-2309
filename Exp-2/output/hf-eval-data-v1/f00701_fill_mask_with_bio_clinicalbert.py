from transformers import AutoTokenizer, AutoModel

def fill_mask_with_bio_clinicalbert(masked_sentence):
    """
    This function fills the masked word in a sentence using the Bio_ClinicalBERT model.
    
    Parameters:
    masked_sentence (str): The sentence with a missing word, represented as [MASK]
    
    Returns:
    str: The sentence with the masked word filled
    """
    # Load the Bio_ClinicalBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    
    # Tokenize the input sentence with the missing word
    input_tokens = tokenizer.encode(masked_sentence, return_tensors="pt")
    
    # Pass the tokenized sentence to the Bio_ClinicalBERT model
    output_logits = model(input_tokens).logits
    
    # Get the most probable word that can fill the mask token
    top_predicted_word = tokenizer.decode(output_logits.argmax(-1).item())
    
    # Replace the mask token with the predicted word in the sentence
    filled_sentence = masked_sentence.replace("[MASK]", top_predicted_word)
    
    return filled_sentence