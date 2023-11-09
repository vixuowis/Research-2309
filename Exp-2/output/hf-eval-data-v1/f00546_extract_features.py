from transformers import AutoTokenizer, AutoModel

# Function to extract features from biomedical entity names using SapBERT model
# Input: biomedical entity names as a string
# Output: [CLS] embedding representing aggregated features of the input entities
def extract_features(entity_names):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

    # Tokenize the input entity names
    inputs = tokenizer(entity_names, return_tensors='pt')

    # Pass the tokenized input to the model
    outputs = model(**inputs)

    # Retrieve the [CLS] embedding from the model output
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding