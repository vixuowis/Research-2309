# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# function_code --------------------

def perform_clinical_ner(text):
    """
    Apply the Bio_ClinicalBERT model to perform NER on medical text.

    Args:
        text (str): The medical text to analyze.

    Returns:
        list: A list of tuples with entities and their labels.

    Raises:
        ValueError: If the input text is not a string or is empty.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("Input text must be a non-empty string.")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = AutoModelForTokenClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1)

    # Extract entities
    entities = []
    for idx, pred in enumerate(predictions[0].tolist()):
        label = model.config.id2label[pred]
        if label != 'O':  # Exclude 'O' labels which represent 'Outside' of named entities
            word = inputs.tokens(idx)
            entities.append((word, label))

    return entities

# test_function_code --------------------

def test_perform_clinical_ner():
    print("Testing started.")

    # Test case 1: Check if function raises a ValueError for invalid input
    invalid_input = ""
    try:
        perform_clinical_ner(invalid_input)
        assert False, "Test case [1/3] failed: Function did not raise ValueError for empty string."
    except ValueError:
        pass

    # Test case 2: Check if function works with valid medical text
    valid_input = "The patient was given 2 doses of Ibuprofen for headache."
    entities = perform_clinical_ner(valid_input)
    assert len(entities) > 0, "Test case [2/3] failed: No entities extracted from valid medical text."

    # Test case 3: Check for specific entity recognition
    for word, label in entities:
        if word.lower() == "ibuprofen" and label.startswith("B-MED"):  # Example entity label
            break
    else:
        assert False, "Test case [3/3] failed: Expected entity not found in the medical text."

    print("Testing finished.")

# call_test_function_line --------------------

test_perform_clinical_ner()