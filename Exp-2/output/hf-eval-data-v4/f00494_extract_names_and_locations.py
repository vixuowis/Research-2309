# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForTokenClassification, AutoTokenizer

# function_code --------------------

def extract_names_and_locations(text):
    """
    Extracts mentioned names and locations from a given text using a pre-trained token classification model.

    :param text: The text from which names and locations are to be extracted.
    :returns: A dictionary with two lists containing the extracted names and locations.
    """
    # Load pre-trained token classification model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained('ismail-lucifer011/autotrain-job_all-903929564', use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained('ismail-lucifer011/autotrain-job_all-903929564', use_auth_token=True)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    # Extract predictions
    predictions = outputs.logits.argmax(dim=2).squeeze().tolist()

    # Convert ids to tokens and labels
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    labels = [model.config.id2label[pred] for pred in predictions]

    # Extract names and locations based on labels
    names = [token for token, label in zip(tokens, labels) if label == 'B-PER' or label == 'I-PER']
    locations = [token for token, label in zip(tokens, labels) if label == 'B-LOC' or label == 'I-LOC']

    return {'names': names, 'locations': locations}


# test_function_code --------------------

