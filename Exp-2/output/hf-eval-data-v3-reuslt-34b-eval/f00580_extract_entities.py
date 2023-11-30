# function_import --------------------

from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# function_code --------------------

def extract_entities(sentence: str) -> dict:
    """
    Extract entities from a provided sentence mentioning various companies and their CEOs.

    Args:
        sentence (str): The sentence from which to extract entities.

    Returns:
        dict: A dictionary with the entities and their types.
    """

    
    # Load trained model for entity extraction
    tokenizer = AutoTokenizer.from_pretrained("./model/tokenizer")
    model = AutoModelForTokenClassification.from_pretrained("./model", return_dict=True)

    # Prepare input
    input = tokenizer(sentence, return_tensors="pt")
    
    # Extract entities using the pre-trained entity extraction model
    output = model(**input)
    predictions = output.logits[0].argmax(-1).numpy()
    predicted_labels = [model.config.id2label[x] for x in predictions][1:-1] # Remove special tokens
    
    # Extract entities from labels and map them to their types (e.g. B-ORG -> ORG)
    entities = []
    last_type = None
    start_idx = -1
    end_idx   = -1
    
    for i, label in enumerate(predicted_labels):
        
        if last_type is not None and ("O" == label or "B" in label):
            entities.append({'entity': sentence[start_idx:end_idx], 'type': last_type})
            
        if label != "O":
            if "B" in label:
                start_idx = i
            else:
                end_idx = i
            last_type = label[-3:] # B-ORG -> ORG, I-PER -> PER
    
    return entities

# test_function_code --------------------

def test_extract_entities():
    """
    Test the extract_entities function.
    """
    sentence1 = "Apple's CEO is Tim Cook and Microsoft's CEO is Satya Nadella"
    sentence2 = "Google's CEO is Sundar Pichai"
    sentence3 = "Amazon's CEO is Andy Jassy"
    assert isinstance(extract_entities(sentence1), dict)
    assert isinstance(extract_entities(sentence2), dict)
    assert isinstance(extract_entities(sentence3), dict)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_extract_entities()