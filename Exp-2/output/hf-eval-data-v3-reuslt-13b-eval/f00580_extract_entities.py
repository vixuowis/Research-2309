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
    
    model = AutoModelForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    tokens = tokenizer.tokenize(sentence)
    inputs = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    outputs = model(inputs)[0]
    
    predictions = torch.argmax(outputs, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs.squeeze().tolist())
    predictions = predictions.squeeze().tolist()
    
    entities = {}
    for prediction in zip(predictions, tokens):
        # print(prediction)
        
        pred, tok = prediction
        if str(pred) == '0':  # O - Outside of an entity.
            continue
                            
        if str(pred) not in entities:
            entities[str(pred)] = []
        entities[str(pred)].append(tok)
    
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