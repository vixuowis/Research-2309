# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features_biobert(text):
    """
    This function uses the BioBERT model from Hugging Face Transformers to extract features from biomedical text.
    BioBERT is a pre-trained biomedical language representation model for biomedical text mining tasks such as biomedical named entity recognition, relation extraction, and question answering.

    Args:
        text (str): The biomedical text from which to extract features.

    Returns:
        torch.Tensor: The extracted features from the BioBERT model.
    """
    model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state

# test_function_code --------------------

def test_extract_features_biobert():
    """
    This function tests the extract_features_biobert function.
    It uses a sample biomedical text and checks if the output is a torch.Tensor.
    """
    sample_text = 'The patient was diagnosed with lung cancer.'
    features = extract_features_biobert(sample_text)
    assert isinstance(features, torch.Tensor), 'The output should be a torch.Tensor.'

# call_test_function_code --------------------

test_extract_features_biobert()