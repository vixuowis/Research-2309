# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def highlight_entities_in_text(text):
    tokenizer = AutoTokenizer.from_pretrained('Jean-Baptiste/camembert-ner')
    model = AutoModelForTokenClassification.from_pretrained('Jean-Baptiste/camembert-ner')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')

    entities = nlp(text)
    highlighted_text = []
    previous_offset = 0
    for entity in entities:
        start_offset, end_offset = entity['start'], entity['end']
        label = entity['entity']
        if label in ['B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']:
            highlighted_text.append(text[previous_offset:start_offset])
            highlighted_text.append(f'[{text[start_offset:end_offset]}]')
            previous_offset = end_offset
    highlighted_text.append(text[previous_offset:])
    return ''.join(highlighted_text)

# test_function_code --------------------

def test_highlight_entities_in_text():
    print("Testing started.")
    sample_text = "La socit de Paris est spcialise dans la vente de vhicules lectriques. Responsable des ventes, vous travaillerez au sein d'une quipe dynamique dans l'agence de Lyon. Vous tes charg(e) de dvelopper le portefeuille client et d'assurer la satisfaction des clients existants."

    # Expected cities and organizations to be highlighted
    expected_result = "La socit de [Paris] est spcialise dans la vente de vhicules lectriques. Responsable des ventes, vous travaillerez au sein d'une quipe dynamique dans l'agence de [Lyon]. Vous tes charg(e) de dvelopper le portefeuille client et d'assurer la satisfaction des clients existants."

    # Testing the function
    print("Testing case [1/1] started.")
    actual_result = highlight_entities_in_text(sample_text)
    assert actual_result == expected_result, f"Test case [1/1] failed: Expected {expected_result}, but got {actual_result}"
    print("Testing finished.")

# Run the test function
test_highlight_entities_in_text()