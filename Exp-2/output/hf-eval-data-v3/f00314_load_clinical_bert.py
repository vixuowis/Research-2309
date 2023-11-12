# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def load_clinical_bert():
    """
    Load the pre-trained model 'emilyalsentzer/Bio_ClinicalBERT' from the transformers library.
    This model is specifically trained on medical data and is ideal for processing and understanding medical reports.
    
    Returns:
        tokenizer: The tokenizer associated with the pre-trained model.
        model: The pre-trained model.
    """
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    return tokenizer, model

# test_function_code --------------------

def test_load_clinical_bert():
    """
    Test the function load_clinical_bert.
    """
    tokenizer, model = load_clinical_bert()
    assert isinstance(tokenizer, AutoTokenizer), 'Tokenizer is not an instance of AutoTokenizer.'
    assert isinstance(model, AutoModel), 'Model is not an instance of AutoModel.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_load_clinical_bert()