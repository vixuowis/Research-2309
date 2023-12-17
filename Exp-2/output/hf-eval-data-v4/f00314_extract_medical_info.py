# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# function_code --------------------

def extract_medical_info(report_text):
    """
    Extracts medical information from the given medical report text
    using a pre-trained `Bio_ClinicalBERT` model.

    Args:
    - report_text (str): A string containing text from a medical report.

    Returns:
    - List of tuples containing extracted entities and their types.
    """
    # Load pre-trained tokenizer and model for token classification
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = AutoModelForTokenClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    # Create a pipeline for named entity recognition (NER)
    ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)

    # Process the report text to extract entities
    entities = ner_pipeline(report_text)

    # Filter to keep only relevant medical information
    relevant_entities = []
    for entity in entities:
        # Usually, medical entities are tagged with labels like 'B-Disease', 'I-Treatment', etc.
        if entity['entity'].startswith('B-') or entity['entity'].startswith('I-'):
            relevant_entities.append((entity['word'], entity['entity']))

    return relevant_entities

# test_function_code --------------------

def test_extract_medical_info():
    print("Testing started.")
    # Prepare a synthetic medical report for testing
    sample_reports = [
        "The patient is diagnosed with acute bronchitis.",
        "Symptoms include headache and muscle pain, indicating flu.",
        "Liver enzymes are elevated, suggesting possible hepatitis."
    ]

    # Test case 1: Acute bronchitis should be recognized
    print("Testing case [1/3] started.")
    assert extract_medical_info(sample_reports[0]), f"Test case [1/3] failed: Acute bronchitis was not recognized."

    # Test case 2: Headache and muscle pain should be recognized
    print("Testing case [2/3] started.")
    assert extract_medical_info(sample_reports[1]), f"Test case [2/3] failed: Headache and muscle pain were not recognized."

    # Test case 3: Elevated liver enzymes should hint at hepatitis
    print("Testing case [3/3] started.")
    assert extract_medical_info(sample_reports[2]), f"Test case [3/3] failed: Possible hepatitis was not recognized."
    print("Testing finished.")