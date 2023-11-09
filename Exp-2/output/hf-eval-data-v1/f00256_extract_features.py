from transformers import PreTrainedTokenizerFast, BartModel

# Function to extract features from Korean text using KoBART model
# @param input_text: The Korean text from which to extract features
# @return: The extracted features

def extract_features(input_text):
    # Load the KoBART-based tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
    # Load the pre-trained KoBART model
    model = BartModel.from_pretrained('gogamza/kobart-base-v2')
    # Tokenize the input text
    tokens = tokenizer(input_text, return_tensors="pt")
    # Pass the tokens to the model to obtain the features
    features = model(**tokens)['last_hidden_state']
    return features