from transformers import BartTokenizer, BartModel

# Function to summarize student essays using BART model
# Input: essay - string, the student's essay
# Output: last_hidden_states - tensor, the last hidden states of the BART model

def summarize_essay(essay):
    # Load the BART tokenizer and model
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartModel.from_pretrained('facebook/bart-base')

    # Convert the essay into tokens compatible with the BART model
    inputs = tokenizer(essay, return_tensors='pt')

    # Pass the tokens into the BART model
    outputs = model(**inputs)

    # Get the last hidden states of the BART model
    last_hidden_states = outputs.last_hidden_state

    return last_hidden_states