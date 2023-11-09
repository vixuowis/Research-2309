from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model using the `from_pretrained` method with the provided model name
# This model is trained to classify text inputs into either questions or statements.
tokenizer = AutoTokenizer.from_pretrained('shahrukhx01/question-vs-statement-classifier')
model = AutoModelForSequenceClassification.from_pretrained('shahrukhx01/question-vs-statement-classifier')

def classify_text(text):
    """
    This function classifies the input text into either a question or a statement.
    
    Parameters:
    text (str): The text to be classified.
    
    Returns:
    str: The classification result - either 'question' or 'statement'.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    # Get the model's output
    outputs = model(**inputs)
    # Get the predicted class
    predicted_class = outputs.logits.argmax(dim=-1).item()
    
    # Return the classification result
    if predicted_class == 0:
        return "question"
    else:
        return "statement"