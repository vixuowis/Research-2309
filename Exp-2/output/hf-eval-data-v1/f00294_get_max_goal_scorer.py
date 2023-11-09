from transformers import TapasForQuestionAnswering, TapasTokenizer

# Function to get the player who scored the maximum goals in a given match
# Uses the TAPAS large model fine-tuned on Sequential Question Answering (SQA) from the transformers library
# The model is pre-trained on MLM and an additional step which the authors call intermediate pre-training, and then fine-tuned on SQA
# It uses relative position embeddings (i.e. resetting the position index at every cell of the table)
def get_max_goal_scorer(question: str, table: str) -> str:
    # Load the pre-trained model
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-large-finetuned-sqa')
    # Load the tokenizer
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-large-finetuned-sqa')
    # Convert the table data and the question into the format required by the model
    inputs = tokenizer(question, table, return_tensors="pt")
    # Pass the processed input to the model and get the answer prediction
    outputs = model(**inputs)
    # Extract the answer from the model's output
    answer_label = tokenizer.convert_ids_to_tokens(outputs.logits.argmax(axis=2)[0, 0])
    return answer_label