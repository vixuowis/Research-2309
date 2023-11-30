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
    
    # Load the pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    model = AutoModelForTokenClassification.from_pretrained("./model/")
    
    # Encode the sentence using BERT tokenizer
    encoding = tokenizer(sentence, return_offsets_mapping=True)
    input_ids = torch.tensor([encoding["input_ids"]])
    attention_mask = torch.tensor([encoding["attention_mask"]])
    
    # Get the predicted tags using our model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        
    labels = np.argmax(outputs[0].numpy(), axis=2)[0]
    
    # Create a mapping between the tagged tokens and the original sentence
    tokens = tokenizer.convert_ids_to_tokens(input_ids.numpy()[0])
    offsets = encoding["offset_mapping"][0]
    tags = [(token, label) for token, label in zip(tokens, labels) if not token.startswith("##")]
    
    # Collect all entities and their types
    entities = {}
    current_entity = ""
    entity_type = "O"
    for item in tags:
        if item[1] != 0:
            if item[1] == 1 or item[1] ==2: #B-CEO/B-CMPY
                if current_entity != "":
                    entities[current_entity] = entity_type
                entity_type = "B" + item[0].split("-")[-1]
                current_entity = ""
            else: #I-CEO/I-CMPY
                if not current_entity or entity_type.endswith(item[0].split("-")[-1]):
                    current_entity += item[0] 
    
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