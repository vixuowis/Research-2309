from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Function to extract company names from text
# This function uses a pre-trained model from Hugging Face Transformers to perform token classification
# The model is trained to identify and extract company names from text

def extract_company_names(text):
    # Load the pre-trained model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained('ismail-lucifer011/autotrain-company_all-903429548', use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained('ismail-lucifer011/autotrain-company_all-903429548', use_auth_token=True)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt')

    # Use the model to analyze the tokenized input and predict the entity tags
    outputs = model(**inputs)

    # Extract the predictions from the outputs
    predictions = torch.argmax(outputs.logits, dim=2)

    # Initialize an empty list to store the company names
    company_names = []

    # Iterate through the predictions
    for prediction in predictions[0]:
        # If the prediction corresponds to the tag for a company name, add the corresponding token to the list of company names
        if prediction == 3:  # Assuming that the tag for a company name is 3
            company_names.append(tokenizer.convert_ids_to_tokens(prediction))

    # Return the list of company names
    return company_names