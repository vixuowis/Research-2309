# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def extract_positional_relations(text):
    """
    Extracts the positional relations between various keywords of a given medical text using the SapBERT model.

    Args:
        text (str): The medical text from which to extract positional relations.

    Returns:
        Tensor: The [CLS] embedding of the last layer of the SapBERT model, which indicates the position of the embedded biomedical entities in the given medical text.
    """
    tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# test_function_code --------------------

def test_extract_positional_relations():
    """
    Tests the extract_positional_relations function by passing a sample medical text and checking the output type.
    """
    sample_text = 'covid infection'
    output = extract_positional_relations(sample_text)
    assert isinstance(output, torch.Tensor), 'Output should be a PyTorch Tensor.'

# call_test_function_code --------------------

test_extract_positional_relations()