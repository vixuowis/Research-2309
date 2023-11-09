from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# Function to classify comments into toxic or non-toxic categories
# This function uses the pre-trained model 'martin-ha/toxic-comment-model' from Hugging Face Transformers
# The model is a fine-tuned DistilBERT model specialized in classifying toxic comments
# The function takes a comment as input and returns the probability of the comment being toxic or non-toxic

def classify_comment(comment):
    model_path = 'martin-ha/toxic-comment-model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return pipeline(comment)