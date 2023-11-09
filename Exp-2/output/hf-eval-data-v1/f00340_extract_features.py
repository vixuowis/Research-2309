from transformers import AutoModel, AutoTokenizer

# Function to extract features from biomedical entity names using the pre-trained model 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
def extract_features(entity):
    '''
    This function takes a biomedical entity name as input and returns the [CLS] embedding of the last layer as output.
    The model used is 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext' from Hugging Face Transformers.
    '''
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

    # Tokenize the input entity and return tensors
    inputs = tokenizer(entity, return_tensors='pt')

    # Get the model outputs
    outputs = model(**inputs)

    # Get the [CLS] embedding of the last layer
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding