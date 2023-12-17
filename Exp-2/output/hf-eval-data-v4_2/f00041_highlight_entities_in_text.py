# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def highlight_entities_in_text(text):
    """
    Highlight names of organizations or cities within the given text using a Named Entity Recognition model.

    Args:
        text (str): The text containing job descriptions in French.

    Returns:
        str: The text with organizations and cities highlighted by being enclosed in square brackets.

    Raises:
        ImportError: If the required transformers library is not installed.
    """
    tokenizer = AutoTokenizer.from_pretrained('Jean-Baptiste/camembert-ner')
    model = AutoModelForTokenClassification.from_pretrained('Jean-Baptiste/camembert-ner')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')

    entities = nlp(text)
    highlighted_text = []
    previous_offset = 0
    for entity in entities:
        start_offset, end_offset = entity['start'], entity['end']
        label = entity['entity_group']
        if label in ['ORG', 'LOC']:
            highlighted_text.append(text[previous_offset:start_offset])
            highlighted_text.append(f'[{text[start_offset:end_offset]}]')
            previous_offset = end_offset
    highlighted_text.append(text[previous_offset:])
    return ''.join(highlighted_text)

# test_function_code --------------------

def test_highlight_entities_in_text():
    print("Testing started.")
    sample_text = "La société de Paris est spécialisée dans la vente de véhicules électriques."

    # Test case 1: Single organization or city
    print("Testing case [1/1] started.")
    highlighted = highlight_entities_in_text(sample_text)
    assert '[Paris]' in highlighted, f"Test case [1/1] failed: expected '[Paris]' in the highlighted text, got {highlighted}"
    print("Testing finished.")

# call_test_function_line --------------------

test_highlight_entities_in_text()