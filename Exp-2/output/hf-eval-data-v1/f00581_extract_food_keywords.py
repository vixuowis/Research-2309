from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Function to extract food keywords from user's input text
# Uses the 'Dizex/InstaFoodRoBERTa-NER' model from Transformers library
# This model is specifically trained for Named Entity Recognition of food items in informal text
# The function takes user input as argument and returns a list of food-related entities

def extract_food_keywords(user_input):
    tokenizer = AutoTokenizer.from_pretrained('Dizex/InstaFoodRoBERTa-NER')
    model = AutoModelForTokenClassification.from_pretrained('Dizex/InstaFoodRoBERTa-NER')
    food_entity_recognition = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')
    food_keywords = food_entity_recognition(user_input)
    return food_keywords