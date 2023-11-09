def test_load_bio_clinical_bert():
    """
    This function tests the 'load_bio_clinical_bert' function by loading the model and checking its type.
    """
    tokenizer, model = load_bio_clinical_bert()
    # Check if the loaded model and tokenizer are of the correct type
    assert isinstance(tokenizer, AutoTokenizer), 'Tokenizer loading failed'
    assert isinstance(model, AutoModel), 'Model loading failed'
    print('All tests passed.')

test_load_bio_clinical_bert()