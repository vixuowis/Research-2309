from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# This function is used to detect if there are any harmful messages in a chat room.
# It uses the Hugging Face Transformers library and a pre-trained model for text classification.
# The model is a fine-tuned version of the DistilBERT model to classify toxic comments.
def detect_toxic_comment(message):
    # Specify the model path
    model_path = 'martin-ha/toxic-comment-model'
    
    # Load the tokenizer and model from the specified path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Create a pipeline for text classification
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    
    # Use the pipeline to classify the harmfulness of the given message
    toxicity_result = pipeline(message)
    
    # Return the classification result
    return toxicity_result