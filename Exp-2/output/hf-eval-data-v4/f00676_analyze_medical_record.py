# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# function_code --------------------

def analyze_medical_record(text):
    """
    Analyze a medical record text and recognize biomedical entities using a pretrained NER model.

    Args:
        text (str): A string containing a medical case report.

    Returns:
        list: A list of identified biomedical entities with their labels.
    """
    tokenizer = AutoTokenizer.from_pretrained('d4data/biomedical-ner-all')
    model = AutoModelForTokenClassification.from_pretrained('d4data/biomedical-ner-all')
    ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)
    results = ner_pipeline(text)
    return results

# test_function_code --------------------

def test_analyze_medical_record():
    print("Testing started.")
    sample_data = "The patient reported no recurrence of palpitations at follow-up 6 months after the ablation."

    # Test case 1
    print("Testing case [1/1] started.")
    results = analyze_medical_record(sample_data)
    assert type(results) is list and len(results) > 0, f"Test case [1/1] failed: Expected non-empty list, got {results}"
    print("Test case [1/1] succeeded.")
    print("Testing finished.")

test_analyze_medical_record()