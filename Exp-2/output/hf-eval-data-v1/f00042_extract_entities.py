from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

# Function to extract entities from text using Named Entity Recognition
# Uses the 'dslim/bert-large-NER' model from Transformers
# Input: A string of text
# Output: A list of entities identified in the text

def extract_entities(text):
    # Load the model
    model = AutoModelForTokenClassification.from_pretrained('dslim/bert-large-NER')
    tokenizer = AutoTokenizer.from_pretrained('dslim/bert-large-NER')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    # Identify entities in the text
    entities = nlp(text)
    return entities