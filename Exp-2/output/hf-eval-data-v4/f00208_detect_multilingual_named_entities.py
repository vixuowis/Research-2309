# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def detect_multilingual_named_entities(text):
    # Initialize the tokenizer and model using the pretrained multilingual NER model
    tokenizer = AutoTokenizer.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    model = AutoModelForTokenClassification.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    
    # Create a pipeline for named entity recognition (NER)
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)

    # Process the input text and return the named entity recognition results
    ner_results = nlp(text)
    return ner_results

# test_function_code --------------------

def test_detect_multilingual_named_entities():
    test_text = 'Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute.'
    results = detect_multilingual_named_entities(test_text)
    assert type(results) is list and len(results) > 0, 'No named entities detected'
    print('Named entity detection test passed.')

test_detect_multilingual_named_entities()