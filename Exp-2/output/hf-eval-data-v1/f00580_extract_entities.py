from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Function to extract entities from a sentence
# Uses the Hugging Face Transformers library and a pre-trained model
# The model is trained for Natural Language Processing Token Classification
# The model is loaded using its identifier 'ismail-lucifer011/autotrain-name_all-904029577'
def extract_entities(sentence):
    # Load the pre-trained model
    model = AutoModelForTokenClassification.from_pretrained('ismail-lucifer011/autotrain-name_all-904029577', use_auth_token=True)
    # Load the corresponding tokenizer
    tokenizer = AutoTokenizer.from_pretrained('ismail-lucifer011/autotrain-name_all-904029577', use_auth_token=True)
    # Tokenize the input sentence and convert to PyTorch tensor
    inputs = tokenizer(sentence, return_tensors='pt')
    # Send the input tokens to the model
    outputs = model(**inputs)
    # Convert the output to a human-readable format
    entities = torch.argmax(outputs.logits, dim=2)
    # Return the extracted entities
    return entities