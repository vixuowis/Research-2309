# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features_russian_text(input_text):
    """
    Extracts features from Russian text using the pre-trained model 'DeepPavlov/rubert-base-cased'.

    Args:
        input_text (str): The Russian text from which to extract features.

    Returns:
        torch.Tensor: The extracted features from the input text.
    """
    tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased')
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state

# test_function_code --------------------

def test_extract_features_russian_text():
    """
    Tests the function 'extract_features_russian_text'.
    """
    input_text = 'Пример текста на русском языке.'
    features = extract_features_russian_text(input_text)
    assert features is not None
    assert features.size(0) == 1

# call_test_function_code --------------------

test_extract_features_russian_text()