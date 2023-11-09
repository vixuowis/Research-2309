from transformers import TapasForQuestionAnswering, TapasTokenizer

# Function to check if the total revenue for last week met our target revenue
# Uses the TAPAS model from Hugging Face Transformers
# Inputs: table - a dictionary with 'Day' and 'Revenue' as keys and lists of days and revenues as values
#         query - a string asking whether the target revenue has been achieved
# Returns: predicted answer coordinates and aggregation indices

def check_revenue_target(table, query):
    # Load the pre-trained model and tokenizer
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-small-finetuned-wtq')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-small-finetuned-wtq')

    # Prepare the inputs for the model
    inputs = tokenizer(table=table, queries=query, return_tensors='pt')

    # Pass the resulting tensors to the model and obtain the logits and logits_aggregation
    outputs = model(**inputs)

    # Convert the logits into the predicted answer coordinates and aggregation indices
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())

    return predicted_answer_coordinates, predicted_aggregation_indices