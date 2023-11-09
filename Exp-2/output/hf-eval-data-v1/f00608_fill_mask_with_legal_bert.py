from transformers import AutoTokenizer, AutoModel

# Function to fill in the gap in a legal document using the 'nlpaueb/legal-bert-small-uncased' model
# This function takes in a string with a '[MASK]' token and returns the filled in text

def fill_mask_with_legal_bert(text):
    # Load the pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-small-uncased')
    model = AutoModel.from_pretrained('nlpaueb/legal-bert-small-uncased')

    # Tokenize the input text
    input = tokenizer.encode(text, return_tensors='pt')

    # Generate predictions
    mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
    token_logits = model(input)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]

    # Get the top 5 tokens
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    # Fill in the mask token with the top predicted token
    for token in top_5_tokens:
        print(f"{text.replace(tokenizer.mask_token, tokenizer.decode([token]))}")