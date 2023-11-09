from transformers import TapasTokenizer, TapasForQuestionAnswering

def get_total_sales(sales_data_table, question):
    '''
    This function uses the TAPAS base model fine-tuned on Sequential Question Answering (SQA) from Hugging Face Transformers to answer questions related to a table in a conversational set-up.
    The function takes a sales data table and a question as input and returns the total sales of a specific product.
    '''
    # Load the pre-trained tokenizer and model
    tokenizer = TapasTokenizer.from_pretrained('lysandre/tapas-temporary-repo')
    model = TapasForQuestionAnswering.from_pretrained('lysandre/tapas-temporary-repo')
    
    # Create input tensors for the given question and sales data table
    inputs = tokenizer(table=sales_data_table, queries=question, return_tensors='pt')
    
    # Pass the input tensors to the model to receive the predicted answer coordinates and aggregation indices
    outputs = model(**inputs)
    
    # Convert the logits to predictions
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())
    
    # Extract the sum of sales for the desired product from the table using the predicted answer coordinates and aggregation indices
    total_sales = 0
    for coordinates in predicted_answer_coordinates:
        for coordinate in coordinates:
            total_sales += sales_data_table[coordinate[0]][coordinate[1]]
    
    return total_sales